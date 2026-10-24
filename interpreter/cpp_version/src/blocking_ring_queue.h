#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

template<typename T>
class BlockingRingQueue {
public:
    enum class PopResult { Item, Empty, Closed };

    explicit BlockingRingQueue(size_t capacity)
        : capacity_(capacity > 0 ? capacity : 1),
          slots_(capacity_),
          head_(0),
          tail_(0),
          count_(0),
          closed_(false) {}

    BlockingRingQueue(const BlockingRingQueue&) = delete;
    BlockingRingQueue& operator=(const BlockingRingQueue&) = delete;

    // Blocks while full. If the queue is closed, drops @p item and returns.
    void push(T item) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_not_full_.wait(lock, [this] { return count_ < capacity_ || closed_; });
            if (closed_) {
                return;
            }
            slots_[tail_] = std::move(item);
            tail_ = (tail_ + 1) % capacity_;
            ++count_;
            cv_not_empty_.notify_one();
        }
        notify_hook();
    }

    // Blocks while empty. Returns false when closed and empty.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [this] { return count_ > 0 || closed_; });
        if (count_ == 0) {
            return false;
        }
        out = std::move(slots_[head_]);
        head_ = (head_ + 1) % capacity_;
        --count_;
        cv_not_full_.notify_one();
        return true;
    }

    // Does not block. Closed is reported only when the queue is also empty.
    PopResult try_pop(T& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == 0) {
            return closed_ ? PopResult::Closed : PopResult::Empty;
        }
        out = std::move(slots_[head_]);
        head_ = (head_ + 1) % capacity_;
        --count_;
        cv_not_full_.notify_one();
        return PopResult::Item;
    }

    // True when a non-blocking pop would not return Empty.
    bool ready() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ > 0 || closed_;
    }

    void close() {
        bool newly = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            newly = !closed_;
            closed_ = true;
            cv_not_empty_.notify_all();
            cv_not_full_.notify_all();
        }
        if (newly) {
            notify_hook();
        }
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    // Invoked without the queue mutex held, after a successful push or the first close.
    // Used to wait on either of two queues. The hook must not call back into this queue.
    void set_event_hook(std::function<void()> hook) {
        std::lock_guard<std::mutex> lock(mutex_);
        event_hook_ = std::move(hook);
    }

private:
    void notify_hook() {
        std::function<void()> hook;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            hook = event_hook_;
        }
        if (hook) {
            hook();
        }
    }

    const size_t capacity_;
    std::vector<T> slots_;
    size_t head_;
    size_t tail_;
    size_t count_;
    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    bool closed_;
    std::function<void()> event_hook_;
};
