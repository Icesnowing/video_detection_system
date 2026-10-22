#ifndef TRACE_LOG_H
#define TRACE_LOG_H

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>

// Suppress every log by defining NLOGS.
enum SDKLogLevel {
    LOG_LEVEL_OFF = -1,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG,
};

inline volatile int g_video_log_level = LOG_LEVEL_DEBUG;

inline FILE* g_trace_log_fp = nullptr;
inline std::mutex g_trace_log_mu;
inline std::string g_trace_log_path;

// video_detect_YYYYMMDD_HHMMSS.log from the local time the file is created.
inline std::string trace_log_make_path() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    char name[64];
    std::snprintf(name, sizeof(name), "video_detect_%04d%02d%02d_%02d%02d%02d.log",
                  tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                  tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
    return name;
}

// Last 20 characters of the filename (not the directory prefix).
inline const char* trace_file_tail(const char* path) {
    if (!path || !path[0]) return "";
    const char* base = path;
    for (const char* p = path; *p; ++p) {
        if (*p == '/' || *p == '\\') base = p + 1;
    }
    const size_t n = std::strlen(base);
    return n > 20 ? base + (n - 20) : base;
}

// Opens path, or video_detect_<local create time>.log when path is empty.
// Later LOG* lines are appended until trace_log_close().
inline bool trace_log_open(const char* path = nullptr) {
    std::lock_guard<std::mutex> lock(g_trace_log_mu);
    if (g_trace_log_fp) return true;
    const std::string chosen = (path && path[0]) ? std::string(path) : trace_log_make_path();
    g_trace_log_path = chosen;
    FILE* fp = std::fopen(chosen.c_str(), "w");
    if (!fp) return false;
    g_trace_log_fp = fp;
    return true;
}

inline void trace_log_close() {
    std::lock_guard<std::mutex> lock(g_trace_log_mu);
    if (!g_trace_log_fp) return;
    std::fflush(g_trace_log_fp);
    std::fclose(g_trace_log_fp);
    g_trace_log_fp = nullptr;
}

inline void trace_log_emit(const char* text) {
    std::lock_guard<std::mutex> lock(g_trace_log_mu);
    if (!g_trace_log_fp || !text) return;
    std::fputs(text, g_trace_log_fp);
    std::fflush(g_trace_log_fp);
}

// File-only line that is not a LOG* record (per-frame latency table).
inline void trace_log_raw(const std::string& line) {
    std::lock_guard<std::mutex> lock(g_trace_log_mu);
    if (!g_trace_log_fp) return;
    std::fputs(line.c_str(), g_trace_log_fp);
    std::fputc('\n', g_trace_log_fp);
    std::fflush(g_trace_log_fp);
}

inline void trace_log_print(const char* level, const char* file, int line, const char* fmt, ...) {
    char body[4096];
    va_list ap;
    va_start(ap, fmt);
    if (std::vsnprintf(body, sizeof(body), fmt ? fmt : "", ap) < 0) {
        body[0] = '\0';
    }
    va_end(ap);

    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) %
                    1000;
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif

    char time_buf[96];
    std::snprintf(time_buf, sizeof(time_buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
                  tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                  tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                  static_cast<int>(ms.count()));

    // [local time][level][filename last 20 chars(line)]:message
    char out[4224];
    std::snprintf(out, sizeof(out), "[%s][%s][%.20s(%03d)]:%s\n",
                  time_buf, level, trace_file_tail(file), line, body);

    trace_log_emit(out);
}

#ifdef NLOGS

#define LOGD(format, ...) ((void)0)
#define LOGI(format, ...) ((void)0)
#define LOGW(format, ...) ((void)0)
#define LOGE(format, ...) ((void)0)

#else

#define LOGD(format, ...)                                                                      \
    do {                                                                                       \
        if (g_video_log_level >= LOG_LEVEL_DEBUG) {                                            \
            trace_log_print("D", __FILE__, __LINE__, format, ##__VA_ARGS__);                   \
        }                                                                                      \
    } while (0)

#define LOGI(format, ...)                                                                      \
    do {                                                                                       \
        if (g_video_log_level >= LOG_LEVEL_INFO) {                                             \
            trace_log_print("I", __FILE__, __LINE__, format, ##__VA_ARGS__);                   \
        }                                                                                      \
    } while (0)

#define LOGW(format, ...)                                                                      \
    do {                                                                                       \
        if (g_video_log_level >= LOG_LEVEL_WARNING) {                                          \
            trace_log_print("W", __FILE__, __LINE__, format, ##__VA_ARGS__);                   \
        }                                                                                      \
    } while (0)

#define LOGE(format, ...)                                                                      \
    do {                                                                                       \
        if (g_video_log_level >= LOG_LEVEL_ERROR) {                                            \
            trace_log_print("E", __FILE__, __LINE__, format, ##__VA_ARGS__);                   \
        }                                                                                      \
    } while (0)

#endif

#endif
