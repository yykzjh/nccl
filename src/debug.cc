/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "nccl_net.h"
#include <ctime>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include "param.h"
#include "param/param.h"
#include <mutex>
#include "os.h"
#include "utils.h"
#include "env.h"
#include <cinttypes>

#define NCCL_DEBUG_RESET_TRIGGERED (-2)

int ncclDebugLevel = -1;
static uint32_t ncclDebugTimestampLevels = 0;     // bitmaps of levels that have timestamps turned on
static char ncclDebugTimestampFormat[256];        // with space for subseconds
static int ncclDebugTimestampSubsecondsStart;     // index where the subseconds starts
static uint64_t ncclDebugTimestampMaxSubseconds;  // Max number of subseconds plus 1, used in duration ratio
static int ncclDebugTimestampSubsecondDigits;     // Number of digits to display
static int pid = -1;
static char hostname[1024];
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = ""; // Global string for the last error in human readable form
uint64_t ncclDebugMask = 0;
FILE* ncclDebugFile = stdout;
static std::mutex  ncclDebugMutex;
static std::chrono::steady_clock::time_point ncclEpoch;
static bool ncclWarnSetDebugInfo = false;

static thread_local int tid = -1;

// clang-format off （局部禁用自动格式化，保住可读性）
// 用 NCCL 的参数框架声明了 NCCL_DEBUG 这个环境变量：包括它的类型、默认值、可接受的字符串取值、以及自动生成的帮助文档。
DEFINE_NCCL_PARAM(ncclParamDebugLevel, ncclDebugLogLevel, NCCL_DEBUG, NCCL_LOG_NONE,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  ncclParamOneOf<ncclDebugLogLevel>(makeOptions(
                    makeOption("VERSION", NCCL_LOG_VERSION, "Prints the NCCL version info only"),
                    makeOption("WARN", NCCL_LOG_WARN, "Prints only messages indicating a fatal error."),
                    makeOption("INFO", NCCL_LOG_INFO, "Prints debug message"),
                    makeOption("ABORT", NCCL_LOG_ABORT, ""),
                    makeOption("TRACE", NCCL_LOG_TRACE, "Prints replayable trace info on all calls")
                  )), "Set debug output level, the option is inclusive for any level that is less verbose than the set value");

DEFINE_NCCL_PARAM(ncclParamDebugSubsys, uint64_t, NCCL_DEBUG_SUBSYS,
                  NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  (ncclParamBitsetOf<ncclDebugLogSubSys, uint64_t>(makeOptions(
                    makeOption("INIT", NCCL_INIT, "NCCL and comm initialization (included in default)"),
                    makeOption("COLL", NCCL_COLL, "Collective operations"),
                    makeOption("P2P", NCCL_P2P, "Peer-to-peer transport"),
                    makeOption("SHM", NCCL_SHM, "Shared memory transport"),
                    makeOption("NET", NCCL_NET, "Network transport"),
                    makeOption("GRAPH", NCCL_GRAPH, "Graph search and topology"),
                    makeOption("TUNING", NCCL_TUNING, "Algorithm tuning"),
                    makeOption("ENV", NCCL_ENV, "Parameter settings by config file, EnvVar or EnvPlugins (included in default)"),
                    makeOption("ALLOC", NCCL_ALLOC, "Device memory allocation"),
                    makeOption("ALLOC_HOST", NCCL_ALLOC_HOST, "Host memory allocation"),
                    makeOption("CALL", NCCL_CALL, "API call tracing"),
                    makeOption("PROXY", NCCL_PROXY, "Proxy thread operations"),
                    makeOption("NVLS", NCCL_NVLS, "NVLink SHARP operations"),
                    makeOption("BOOTSTRAP", NCCL_BOOTSTRAP, "Bootstrap network (included in default)"),
                    makeOption("REG", NCCL_REG, "Buffer registration"),
                    makeOption("PROFILE", NCCL_PROFILE,   "Profiling"),
                    makeOption("RAS", NCCL_RAS, "Reliability, availability, serviceability"),
                    makeOption("DESTROY", NCCL_DESTROY, "Communicator destroy, abort, revoke, and plugin unload/close operations"),
                    makeOption("ALL", NCCL_ALL, "All categories")
                  ))), "Filter debug output by (comma-separated)");

DEFINE_NCCL_PARAM(ncclParamWarnEnableDebugInfo, bool, NCCL_WARN_ENABLE_DEBUG_INFO, false,
                  NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "If enabled, the debug level will be set to INFO after a WARN level debug message is logged.");
// clang-format on

DEFINE_NCCL_PARAM(ncclParamDebugTimestampLevel, uint32_t, NCCL_DEBUG_TIMESTAMP_LEVELS, (1u << NCCL_LOG_WARN),
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  ncclParamBitsetOf<uint32_t>(
                    makeOptions(makeOption("VERSION", (1u << NCCL_LOG_VERSION), "on NCCL version info message"),
                                makeOption("WARN", (1u << NCCL_LOG_WARN), "on Explicit error message"),
                                makeOption("INFO", (1u << NCCL_LOG_INFO), "on Debug message"),
                                makeOption("ABORT", (1u << NCCL_LOG_ABORT), ""),
                                makeOption("TRACE", (1u << NCCL_LOG_TRACE), "on Replayable trace message"),
                                makeOption("ALL",
                                           (1u << NCCL_LOG_VERSION | 1u << NCCL_LOG_WARN | 1u << NCCL_LOG_INFO |
                                            1u << NCCL_LOG_ABORT | 1u << NCCL_LOG_TRACE),
                                           "on All messages"))),
                  "Set which log lines get a timestamp depending upon the level of the log");

DEFINE_NCCL_PARAM(ncclParamDebugTsFormat, const char*, NCCL_DEBUG_TIMESTAMP_FORMAT, "[%F %T] ",
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "Set the format used when printing debug log messages");

DEFINE_NCCL_PARAM(ncclParamDebugFile, const char*, NCCL_DEBUG_FILE, nullptr,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "Set the NCCL debug logging output to a file. The filename format can be set to "
                  "filename.%h.%p where %h is replaced with the hostname "
                  "and %p is replaced "
                  "with the process PID. This does not accept the ~ character as part of the path, "
                  "please convert to a relative or absolute path first.");

typedef const char* (*ncclGetEnvFunc_t)(const char*);

static ncclResult_t getHostNameForLog(char* hostname, int maxlen, const char delim) {
  // 获取主机名，不能超过 maxlen 并且被 delim 截断
  ncclResult_t ret = getHostName(hostname, maxlen, delim);
  if (ret != ncclSuccess) return ret;

  // % 和 / 字符都替换为 -
  for (int i = 0; i < maxlen - 1 && hostname[i]; ++i) {
    // Replace special characters in hostnames with dashes
    switch (hostname[i]) {
    case '%':
    case '/':
      hostname[i] = '-';
      break;
    default:
      break;
    }
  }
  return ncclSuccess;
}

// This function must be called with ncclDebugLock locked!
static void ncclDebugInit() {
  int tempNcclDebugLevel = -1;
  // 由于NCCL运行时可能会动态触发日志系统的初始化，所以先判断当前是否是重置状态，如果是重置状态并且之前有打开的日志文件，需要fclose，重置为stdout。
  if (ncclDebugLevel == NCCL_DEBUG_RESET_TRIGGERED && ncclDebugFile != stdout) {
    // Finish the reset initiated via ncclResetDebugInit().
    fclose(ncclDebugFile);
    ncclDebugFile = stdout;
  }

  // 获取 NCCL_DEBUG_LEVEL 环境变量的值
  tempNcclDebugLevel = ncclParamDebugLevel();

  // 获取 NCCL_WARN_ENABLE_DEBUG_INFO 环境变量的值，决定是否在收到 WARN 级别日志后将 debug 级别设置为 INFO
  ncclWarnSetDebugInfo = ncclParamWarnEnableDebugInfo();

  // 获取 NCCL_DEBUG_TIMESTAMP_LEVELS 环境变量的值，决定哪些日志级别会带有时间戳
  // Determine which debug levels will have timestamps.
  ncclDebugTimestampLevels = ncclParamDebugTimestampLevel();

  // 获取 NCCL_DEBUG_TIMESTAMP_FORMAT 环境变量的值，决定时间戳的格式
  // Store a copy of the timestamp format with space for the subseconds, if used.
  const char* tsFormat = ncclParamDebugTsFormat();
  ncclDebugTimestampSubsecondsStart = -1;
  // Find where the subseconds are in the format.
  for (int i = 0; tsFormat[i] != '\0'; ++i) {
    // 两个 %% 转义，跳过第一个
    if (tsFormat[i] == '%' && tsFormat[i + 1] == '%') {
      // Next two chars are "%"
      // Skip the next character, too, and restart checking after that.
      ++i;
      continue;
    }
    // 自定义亚秒部分 %Xf，X 在 1 到 9 之间
    if (tsFormat[i] == '%' &&                               // Found a percentage
        ('1' <= tsFormat[i + 1] && tsFormat[i + 1] <= '9') && // Next char is a digit between 1 and 9 inclusive
        tsFormat[i + 2] == 'f'                                // Two characters later is an "f"
    ) {
      constexpr int replaceLen = sizeof("%Xf") - 1;
      ncclDebugTimestampSubsecondDigits = tsFormat[i + 1] - '0';
      // 指定格式字符串需要的时间戳长度大于最大长度，fall back on the default.
      if (ncclDebugTimestampSubsecondDigits + strlen(tsFormat) - replaceLen > sizeof(ncclDebugTimestampFormat) - 1) {
        // Won't fit; fall back on the default.
        break;
      }
      ncclDebugTimestampSubsecondsStart = i;
      ncclDebugTimestampMaxSubseconds = 1;

      // 将时间戳格式字符串当前之前的部分 copy 到 ncclDebugTimestampFormat
      memcpy(ncclDebugTimestampFormat, tsFormat, i);
      // 再往后添加 X 个空格，ncclDebugTimestampMaxSubseconds 变为以 10 为底的 X 次方
      for (int j = 0; j < ncclDebugTimestampSubsecondDigits; ++j) {
        ncclDebugTimestampFormat[i + j] = ' ';
        ncclDebugTimestampMaxSubseconds *= 10;
      }
      // 时间戳格式字符串后面部分 copy 到 ncclDebugTimestampFormat
      strcpy(ncclDebugTimestampFormat + i + ncclDebugTimestampSubsecondDigits, tsFormat + i + replaceLen);
      break;
    }
  }
  // 没有设置 NCCL 自定义亚秒部分%Xf
  if (ncclDebugTimestampSubsecondsStart == -1) {
    if (strlen(tsFormat) < sizeof(ncclDebugTimestampFormat)) {
      strcpy(ncclDebugTimestampFormat, tsFormat);
    } else {
      strcpy(ncclDebugTimestampFormat, "[%F %T] ");
    }
  }

  // 将时间戳格式字符串中的下划线替换为空格
  // Replace underscore with spaces... it is hard to put spaces in command line parameters.
  for (int i = 0; ncclDebugTimestampFormat[i] != '\0'; ++i) {
    if (ncclDebugTimestampFormat[i] == '_') ncclDebugTimestampFormat[i] = ' ';
  }

  // 获取最长不超过 1024 并且由第一个 . 截断的主机名
  // Cache pid and hostname
  getHostNameForLog(hostname, 1024, '.');
  pid = ncclOsGetPid();

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
  // 获取 NCCL_DEBUG_FILE 环境变量的值，决定日志文件的路径
  const char* ncclDebugFileEnv = ncclParamDebugFile();
  if (tempNcclDebugLevel > NCCL_LOG_VERSION && ncclDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX + 1] = "";
    char* dfn = debugFn;
    // 解析匹配日志文件路径命名中的 %h 为主机名，%p 为进程 ID，其他原样解析
    while (ncclDebugFileEnv[c] != '\0' && (dfn - debugFn) < PATH_MAX) {
      if (ncclDebugFileEnv[c++] != '%') {
        *dfn++ = ncclDebugFileEnv[c - 1];
        continue;
      }
      switch (ncclDebugFileEnv[c++]) {
      case '%': // Double %
        *dfn++ = '%';
        break;
      case 'h': // %h = hostname
        dfn += snprintf(dfn, PATH_MAX + 1 - (dfn - debugFn), "%s", hostname);
        break;
      case 'p': // %p = pid
        dfn += snprintf(dfn, PATH_MAX + 1 - (dfn - debugFn), "%d", pid);
        break;
      default: // Echo everything we don't understand
        *dfn++ = '%';
        if ((dfn - debugFn) < PATH_MAX) {
          *dfn++ = ncclDebugFileEnv[c - 1];
        }
        break;
      }
      if ((dfn - debugFn) > PATH_MAX) {
        // snprintf wanted to overfill the buffer: set dfn to the end
        // of the buffer (for null char) and it will naturally exit
        // the loop.
        dfn = debugFn + PATH_MAX;
      }
    }
    *dfn = '\0';
    // 日志文件是非空路径，打开
    if (debugFn[0] != '\0') {
      FILE* file = fopen(debugFn, "w");
      if (file != nullptr) {
        // fopen 默认为块缓冲，设置为行缓冲
#if defined(NCCL_OS_LINUX)
        setlinebuf(file); // disable block buffering
#elif defined(NCCL_OS_WINDOWS)
        setvbuf(file, NULL, _IOLBF, 0); // disable block buffering
#endif
        ncclDebugFile = file;
      }
    }
  }

  ncclEpoch = std::chrono::steady_clock::now();
  ncclDebugMask = ncclParamDebugSubsys();
  COMPILER_ATOMIC_STORE(&ncclDebugLevel, tempNcclDebugLevel, std::memory_order_release);
}

static void ncclDebugLogV(ncclDebugLogLevel level, unsigned long flags, const char* file, const char* func, int line,
                          const char* fmt, va_list vargs) {
  // 快速路径获取一次无锁快照，避免在多线程环境下获取的值不一致
  int gotLevel = COMPILER_ATOMIC_LOAD(&ncclDebugLevel, std::memory_order_acquire);

  // 线程级别，静默模式 WARN 级别日志降级到 INFO，并且指定子系统
  if (ncclDebugNoWarn != 0 && level == NCCL_LOG_WARN) {
    level = NCCL_LOG_INFO;
    flags = ncclDebugNoWarn;
  }

  // 全局级别，记录最近一条 WARN 级别日志。在 WARN 降级之后，日志过滤之前
  // Save the last error (WARN) as a human readable string
  if (level == NCCL_LOG_WARN) {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    va_list vcopy;
    va_copy(vcopy, vargs);
    (void)vsnprintf(ncclLastError, sizeof(ncclLastError), fmt, vcopy);
    va_end(vcopy);
  }

  // 日志过滤
  if (gotLevel >= 0 && (gotLevel < level || (flags & ncclDebugMask) == 0)) {
    return;
  }

  // 加锁
  std::lock_guard<std::mutex> lock(ncclDebugMutex);
  // 懒加载，加锁后直接获取懒加载可能发生变化的 ncclDebugLevel，gotLevel 已过期
  // 初始化如下变量：
  // - ncclWarnSetDebugInfo
  // - ncclDebugTimestampLevels
  // - ncclDebugTimestampSubsecondsStart
  // - ncclDebugTimestampMaxSubseconds
  // - ncclDebugTimestampFormat
  // - ncclDebugTimestampSubsecondDigits
  // - ncclDebugFile
  if (ncclDebugLevel < 0) ncclDebugInit();
  // 快速路径的过滤可能失效，再次过滤
  if (ncclDebugLevel < level || ((flags & ncclDebugMask) == 0)) {
    return;
  }

  // 当前线程局部懒缓存线程号
  if (tid == -1) {
    tid = ncclOsGetTid();
  }

  char buffer[1024];
  size_t len = 0;

  // WARNs come with an extra newline at the beginning.
  if (level == NCCL_LOG_WARN) {
    buffer[len++] = '\n';
  }

  // 当前日志级别启用了时间戳 Mask
  // Add the timestamp to the buffer if they are turned on for this level.
  if (ncclDebugTimestampLevels & (1 << level)) {
    if (ncclDebugTimestampFormat[0] != '\0') {
      // 获取当前时间秒级时间戳和剩余的纳秒级时间戳
      struct timespec ts;
      clockRealtime(&ts);
      time_t nowTimeT = ts.tv_sec;
      long nowNs = ts.tv_nsec;
      // 秒级时间戳转成日期格式，但 m_year 要 +1900，tm_mon 要 +1
      std::tm nowTm;
      ncclOsLocaltime(&nowTimeT, &nowTm);

      // Add the subseconds portion if it is part of the format.
      char localTimestampFormat[sizeof(ncclDebugTimestampFormat)];
      const char* pformat = ncclDebugTimestampFormat;
      // 如果要求高精度时间，把纳秒级时间戳根据精度填充到之前留出的空格中
      if (ncclDebugTimestampSubsecondsStart != -1) {
        pformat = localTimestampFormat;   // Need to use the local version which has subseconds
        memcpy(localTimestampFormat, ncclDebugTimestampFormat, ncclDebugTimestampSubsecondsStart);
        /*
        %：格式说明开始
        0：宽度不足时在左侧补 0
        *：字段宽度由后续的 int 参数指定
        PRIu64：打印 uint64_t 无符号十进制整数的跨平台宏，定义于 <cinttypes>
        */
        snprintf(localTimestampFormat + ncclDebugTimestampSubsecondsStart, ncclDebugTimestampSubsecondDigits + 1,
                 "%0*" PRIu64, ncclDebugTimestampSubsecondDigits,
                 (uint64_t)(nowNs / (1000000000L / ncclDebugTimestampMaxSubseconds)));
        // 末尾会填充 '\0'
        strcpy(localTimestampFormat + ncclDebugTimestampSubsecondsStart + ncclDebugTimestampSubsecondDigits,
               ncclDebugTimestampFormat + ncclDebugTimestampSubsecondsStart + ncclDebugTimestampSubsecondDigits);
      }

      // 将当前时间格式化为指定格式并添加到 buffer 后
      // Format the time. If it runs out of space, fall back on a simpler format.
      int adv = std::strftime(buffer + len, sizeof(buffer) - len, pformat, &nowTm);
      if (adv == 0 && ncclDebugTimestampFormat[0] != '\0') {
        // Ran out of space. Fall back on the default. This should never fail.
        adv = std::strftime(buffer + len, sizeof(buffer) - len, "[%F %T] ", &nowTm);
      }
      len += adv;
    }
  }
  len = std::min(len, sizeof(buffer) - 1);  // prevent overflows

  // 往后再写入 主机名:进程号:线程号
  // Add hostname, pid and tid portion of the log line.
  if (level != NCCL_LOG_VERSION) {
    len += snprintf(buffer + len, sizeof(buffer) - len, "%s:%d:%d ", hostname, pid, tid);
    len = std::min(len, sizeof(buffer) - 1);  // prevent overflows
  }

  // 获取当前 CUDA 设备号
  int cudaDev = 0;
  if (!(level == NCCL_LOG_TRACE && flags == NCCL_CALL)) {
    (void)cudaGetDevice(&cudaDev);
  }

  // 初始化文件名和函数名字符串
  const char* fileStr = file ? file : "<unknown>";
  const char* funcStr = func ? func : "<unknown>";

  // 根据不同的日志级别组织前缀，这里会把日志传入的 fmt 写入前缀，但没解析
  // Add level specific formatting. The format string from the call site is incorporated into this prefix.
  if (level == NCCL_LOG_WARN) {
    if (func && func[0]) {
      len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] %s:%d (%s) NCCL WARN %s\n", cudaDev, fileStr, line,
                      funcStr, fmt);
    } else {
      len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] %s:%d NCCL WARN %s\n", cudaDev, fileStr, line, fmt);
    }
    if (ncclWarnSetDebugInfo) {
      COMPILER_ATOMIC_STORE(&ncclDebugLevel, static_cast<int>(NCCL_LOG_INFO), std::memory_order_release);
    }
  } else if (level == NCCL_LOG_INFO) {
    len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] NCCL INFO %s\n", cudaDev, fmt);
  } else if (level == NCCL_LOG_TRACE && flags == NCCL_CALL) {
    len += snprintf(buffer + len, sizeof(buffer) - len, "NCCL CALL %s\n", fmt);
  } else if (level == NCCL_LOG_TRACE) {
    auto delta = std::chrono::steady_clock::now() - ncclEpoch;
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count() * 1000;
    len += snprintf(buffer + len, sizeof(buffer) - len, "[%d] %f %s:%d NCCL TRACE %s\n", cudaDev, timestamp, funcStr,
                    line, fmt);
  } else {
    len += snprintf(buffer + len, sizeof(buffer) - len, "%s\n", fmt);
  }

  // 如果发生溢出，行末添加的 '\n' 被截断。由于前面的 snprintf 会确保最后一位是 ‘\0’，最终确保被截断的日志结尾为 \n\0
  // If the prefixed format string overflows, make sure it is still terminated with a newline.
  if (len > sizeof(buffer) - 1) {
    // snprintf already placed a \0 at sizeof(buffer)-1
    buffer[sizeof(buffer) - 2] = '\n';
  }

  // 准备好的日志格式化后写入文件
  // Add the message as given by the call site.
  // The call site's format string has been incorporated into `buffer` along with our prefix.
  va_list vcopy;
  va_copy(vcopy, vargs);
  (void)vfprintf(ncclDebugFile, buffer, vcopy);
  va_end(vcopy);
}

// Internal only Common logging function used by the INFO, WARN and TRACE macros
void ncclDebugLogInternal(ncclDebugLogLevel level, unsigned long flags, const char* file, const char* func, int line,
                          const char* fmt, ...) {
  va_list vargs;
  va_start(vargs, fmt);
  ncclDebugLogV(level, flags, file, func, line, fmt, vargs);
  va_end(vargs);
}

/* Exported ABI logging function exported to the dynamically loadable Net
 * transport modules so they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line, const char* fmt, ...) {
  va_list vargs;
  va_start(vargs, fmt);
  const char* file = nullptr;
  const char* func = nullptr;
  if (level == NCCL_LOG_WARN) {
    file = filefunc;
  } else if (level == NCCL_LOG_TRACE) {
    func = filefunc;
  }
  ncclDebugLogV(level, flags, file, func, line, fmt, vargs);
  va_end(vargs);
}

// Non-deprecated version for internal use.
extern "C"
#if !defined(NCCL_OS_WINDOWS)
  __attribute__((visibility("default")))
#endif
  void ncclResetDebugInitInternal() {
  // Cleans up from a previous ncclDebugInit() and reruns.
  // Use this after changing NCCL_DEBUG and related parameters in the environment.
  std::lock_guard<std::mutex> lock(ncclDebugMutex);
  // Let ncclDebugInit() know to complete the reset.
  COMPILER_ATOMIC_STORE(&ncclDebugLevel, static_cast<int>(NCCL_DEBUG_RESET_TRIGGERED), std::memory_order_release);
}

// In place of: NCCL_API(void, ncclResetDebugInit);
#ifdef pncclResetDebugInit
#undef pncclResetDebugInit
#endif
#if defined(NCCL_OS_LINUX)
__attribute__((visibility("default"))) __attribute__((alias("ncclResetDebugInit")))
#endif
void pncclResetDebugInit();
extern "C"
#if defined(__GNUC__) || defined(__clang__)
  __attribute__((visibility("default"))) __attribute__((weak)) __attribute__((
    deprecated("ncclResetDebugInit is not supported as part of the NCCL API and will be removed in the future")))
#endif
  void ncclResetDebugInit();

extern "C" void ncclResetDebugInit() {
  // This is now deprecated as part of the NCCL API. It will be removed
  // from the API in the future. It is still available as an
  // exported symbol.
  ncclResetDebugInitInternal();
}

DEFINE_NCCL_PARAM(ncclParamSetThreadName, bool, NCCL_SET_THREAD_NAME, false,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT,
                  "Allow NCCL to give meaningful names to NCCL CPU threads via pthread_setname_np");

void ncclSetThreadName(std::thread& thread, const char* fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  // NCCL_SET_THREAD_NAME 决定是否设置线程名
  if (ncclParamSetThreadName() == false) return;
  char threadName[NCCL_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, NCCL_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  // thread.native_handle() 拿到 pthread 句柄
  // pthread_setname_np 设置线程名
  pthread_setname_np(thread.native_handle(), threadName);
#endif
}
