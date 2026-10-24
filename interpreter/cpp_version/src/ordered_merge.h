#pragma once

#include "blocking_ring_queue.h"
#include "frame_packet.h"
#include "trace_log.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

/**
 * Merge Q_skip and Q_infer into one frame_idx sequence.
 *
 * frame_idx is contiguous from 0. Each frame is on exactly one lane, and each
 * lane is FIFO. The next output is whichever head equals next_idx_.
 *
 * An infer frame at next_idx_ is emitted without waiting for a later skip.
 * A skip at next_idx_ is emitted without waiting for a later detection.
 *
 * Skips that arrive before an older in-flight detection are pulled into skip_buf_
 * so the bounded Q_skip cannot stall the preprocess thread (it is also the
 * producer of Q_det). Q_infer keeps a single hold; it is emitted or compared
 * as soon as it is the next frame, so sample_mode=none does not wait for EOS.
 */
class OrderedLaneMerger {
public:
    OrderedLaneMerger(BlockingRingQueue<FramePacket>& skip_lane,
                      BlockingRingQueue<FramePacket>& infer_lane)
        : wake_(std::make_shared<Wakeup>()),
          skip_lane_(skip_lane),
          infer_lane_(infer_lane) {
        auto wake = wake_;
        std::function<void()> hook = [wake] {
            std::lock_guard<std::mutex> lock(wake->mu);
            wake->cv.notify_one();
        };
        skip_lane_.set_event_hook(hook);
        infer_lane_.set_event_hook(hook);
    }

    ~OrderedLaneMerger() {
        skip_lane_.set_event_hook(nullptr);
        infer_lane_.set_event_hook(nullptr);
    }

    OrderedLaneMerger(const OrderedLaneMerger&) = delete;
    OrderedLaneMerger& operator=(const OrderedLaneMerger&) = delete;

    /**
     * Returns true and sets @p out to the next frame in order, or a single eos
     * packet when both lanes are finished. Returns false on duplicate frame_idx,
     * a gap, or a lane that goes backwards.
     */
    bool pop_next(FramePacket& out) {
        if (finished_) {
            return false;
        }

        for (;;) {
            if (!drain_skip() || !poll_infer()) {
                return false;
            }

            if (skip_closed_ && infer_closed_ && skip_buf_.empty() && !infer_hold_) {
                out = FramePacket::make_eos();
                finished_ = true;
                return true;
            }

            if (skip_buf_.size() && infer_hold_ &&
                skip_buf_.front().frame_idx == infer_hold_->frame_idx) {
                LOGE("Ordered merge: duplicate frame_idx %u on skip and infer lanes",
                     skip_buf_.front().frame_idx);
                return false;
            }

            if (skip_buf_.size() && skip_buf_.front().frame_idx == next_idx_) {
                return emit(out, take_skip());
            }
            if (infer_hold_ && infer_hold_->frame_idx == next_idx_) {
                FramePacket packet = std::move(*infer_hold_);
                infer_hold_.reset();
                return emit(out, std::move(packet));
            }

            if (skip_buf_.size() && skip_buf_.front().frame_idx < next_idx_) {
                LOGE("Ordered merge: skip frame_idx %u is behind next %u",
                     skip_buf_.front().frame_idx, next_idx_);
                return false;
            }
            if (infer_hold_ && infer_hold_->frame_idx < next_idx_) {
                LOGE("Ordered merge: infer frame_idx %u is behind next %u",
                     infer_hold_->frame_idx, next_idx_);
                return false;
            }

            // Both heads are past next_idx_, or the only open lane is past it.
            // A frame still inside preprocess/infer is possible only when that
            // lane is open and has not presented a newer head.
            const bool skip_past =
                skip_closed_ || (skip_buf_.size() && skip_buf_.front().frame_idx > next_idx_);
            const bool infer_past =
                infer_closed_ || (infer_hold_ && infer_hold_->frame_idx > next_idx_);
            if (skip_past && infer_past) {
                LOGE("Ordered merge: missing frame_idx %u", next_idx_);
                return false;
            }

            wait_for_input();
        }
    }

private:
    struct Wakeup {
        std::mutex mu;
        std::condition_variable cv;
    };

    bool drain_skip() {
        if (skip_closed_) {
            return true;
        }
        for (;;) {
            FramePacket packet;
            const auto status = skip_lane_.try_pop(packet);
            if (status == BlockingRingQueue<FramePacket>::PopResult::Empty) {
                return true;
            }
            if (status == BlockingRingQueue<FramePacket>::PopResult::Closed) {
                skip_closed_ = true;
                return true;
            }
            if (packet.eos) {
                skip_closed_ = true;
                return true;
            }
            if (!skip_buf_.empty() && packet.frame_idx <= skip_buf_.back().frame_idx) {
                LOGE("Ordered merge: non-increasing frame_idx %u on skip lane", packet.frame_idx);
                return false;
            }
            skip_buf_.push_back(std::move(packet));
        }
    }

    bool poll_infer() {
        if (infer_hold_ || infer_closed_) {
            return true;
        }
        FramePacket packet;
        const auto status = infer_lane_.try_pop(packet);
        if (status == BlockingRingQueue<FramePacket>::PopResult::Empty) {
            return true;
        }
        if (status == BlockingRingQueue<FramePacket>::PopResult::Closed) {
            infer_closed_ = true;
            return true;
        }
        if (packet.eos) {
            infer_closed_ = true;
            return true;
        }
        infer_hold_ = std::move(packet);
        return true;
    }

    FramePacket take_skip() {
        FramePacket packet = std::move(skip_buf_.front());
        skip_buf_.pop_front();
        return packet;
    }

    bool emit(FramePacket& out, FramePacket&& packet) {
        if (has_last_ && packet.frame_idx <= last_idx_) {
            LOGE("Ordered merge: frame_idx %u is not greater than previous %u",
                 packet.frame_idx, last_idx_);
            return false;
        }
        last_idx_ = packet.frame_idx;
        has_last_ = true;
        ++next_idx_;
        out = std::move(packet);
        return true;
    }

    bool input_available() const {
        if (!skip_closed_ && skip_lane_.ready()) {
            return true;
        }
        if (!infer_closed_ && !infer_hold_ && infer_lane_.ready()) {
            return true;
        }
        return false;
    }

    void wait_for_input() {
        std::unique_lock<std::mutex> lock(wake_->mu);
        wake_->cv.wait(lock, [this] { return input_available(); });
    }

    std::shared_ptr<Wakeup> wake_;
    BlockingRingQueue<FramePacket>& skip_lane_;
    BlockingRingQueue<FramePacket>& infer_lane_;
    std::deque<FramePacket> skip_buf_;
    std::optional<FramePacket> infer_hold_;
    bool skip_closed_ = false;
    bool infer_closed_ = false;
    bool finished_ = false;
    uint32_t next_idx_ = 0;
    uint32_t last_idx_ = 0;
    bool has_last_ = false;
};
