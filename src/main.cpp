#include "LIVMapper.h"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <unistd.h>

namespace
{
std::string makeTerminalLogPath()
{
  const std::filesystem::path log_dir = std::filesystem::path(ROOT_DIR) / "Log" / "terminalLog";
  std::error_code error;
  std::filesystem::create_directories(log_dir, error);
  if (error)
  {
    std::fprintf(stderr, "Failed to create terminal log directory '%s': %s\n", log_dir.c_str(), error.message().c_str());
  }

  const auto now = std::chrono::system_clock::now();
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
  localtime_r(&time, &local_time);

  std::ostringstream filename;
  filename << "terminal_output_" << std::put_time(&local_time, "%Y%m%d_%H%M%S") << '_' << std::setfill('0') << std::setw(3)
           << milliseconds.count() << ".log";
  return (log_dir / filename.str()).string();
}

// Capture the process-level stdout/stderr stream in one file while preserving
// the original terminal output. This also covers printf, ROS console messages,
// and output from child processes that inherit these file descriptors.
class TerminalLogger
{
public:
  explicit TerminalLogger(const std::string &log_path)
  {
    log_fd_ = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    terminal_fd_ = ::dup(STDOUT_FILENO);

    int pipe_fds[2] = {-1, -1};
    if (log_fd_ < 0 || terminal_fd_ < 0 || ::pipe(pipe_fds) != 0)
    {
      std::fprintf(stderr, "Failed to initialize terminal logger '%s': %s\n", log_path.c_str(), std::strerror(errno));
      closeDescriptors();
      return;
    }

    read_fd_ = pipe_fds[0];
    const int write_fd = pipe_fds[1];
    std::fflush(stdout);
    std::fflush(stderr);

    if (::dup2(write_fd, STDOUT_FILENO) < 0 || ::dup2(write_fd, STDERR_FILENO) < 0)
    {
      std::fprintf(stderr, "Failed to redirect terminal output: %s\n", std::strerror(errno));
      ::close(write_fd);
      closeDescriptors();
      return;
    }
    ::close(write_fd);

    // Keep newline-terminated printf/ROS messages visible without block-buffer delay.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    active_ = true;
    worker_ = std::thread(&TerminalLogger::copyOutput, this);
    std::cout << "[ Logger ] Terminal output is appended to: " << log_path << std::endl;
  }

  ~TerminalLogger()
  {
    if (!active_)
    {
      closeDescriptors();
      return;
    }

    std::cout.flush();
    std::cerr.flush();
    std::fflush(stdout);
    std::fflush(stderr);

    // Restoring both descriptors closes the last pipe writers and lets the
    // worker drain all pending output before it exits.
    ::dup2(terminal_fd_, STDOUT_FILENO);
    ::dup2(terminal_fd_, STDERR_FILENO);
    if (worker_.joinable()) worker_.join();
    active_ = false;
    closeDescriptors();
  }

  TerminalLogger(const TerminalLogger &) = delete;
  TerminalLogger &operator=(const TerminalLogger &) = delete;

private:
  static void writeAll(int fd, const char *data, size_t size)
  {
    while (size > 0)
    {
      const ssize_t written = ::write(fd, data, size);
      if (written > 0)
      {
        data += written;
        size -= static_cast<size_t>(written);
      }
      else if (written < 0 && errno == EINTR)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }

  void copyOutput()
  {
    std::array<char, 8192> buffer{};
    while (true)
    {
      const ssize_t count = ::read(read_fd_, buffer.data(), buffer.size());
      if (count > 0)
      {
        writeAll(terminal_fd_, buffer.data(), static_cast<size_t>(count));
        writeAll(log_fd_, buffer.data(), static_cast<size_t>(count));
      }
      else if (count < 0 && errno == EINTR)
      {
        continue;
      }
      else
      {
        break;
      }
    }
  }

  void closeDescriptors()
  {
    if (read_fd_ >= 0) ::close(read_fd_);
    if (terminal_fd_ >= 0) ::close(terminal_fd_);
    if (log_fd_ >= 0) ::close(log_fd_);
    read_fd_ = terminal_fd_ = log_fd_ = -1;
  }

  int read_fd_ = -1;
  int terminal_fd_ = -1;
  int log_fd_ = -1;
  bool active_ = false;
  std::thread worker_;
};
} // namespace

int main(int argc, char **argv)
{
  TerminalLogger terminal_logger(makeTerminalLogPath());
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  LIVMapper mapper(nh); 
  mapper.initializeSubscribersAndPublishers(nh, it);
  mapper.run();
  return 0;
}
