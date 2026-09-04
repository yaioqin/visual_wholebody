#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <ros/ros.h>

#include <b2z1_real/B2LowCommand.h>
#include <b2z1_real/B2LowState.h>

#include <unitree/common/thread/thread.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace {

constexpr char kLowCommandTopic[] = "rt/lowcmd";
constexpr char kLowStateTopic[] = "rt/lowstate";
constexpr float kPositionStop = 2.146E9F;
constexpr float kVelocityStop = 16000.0F;

uint32_t Crc32Core(uint32_t* pointer, uint32_t length) {
  uint32_t crc = 0xFFFFFFFF;
  constexpr uint32_t polynomial = 0x04c11db7;
  for (uint32_t index = 0; index < length; ++index) {
    uint32_t xbit = 1U << 31;
    const uint32_t data = pointer[index];
    for (uint32_t bit = 0; bit < 32; ++bit) {
      if ((crc & 0x80000000U) != 0U) {
        crc = (crc << 1U) ^ polynomial;
      } else {
        crc <<= 1U;
      }
      if ((data & xbit) != 0U) {
        crc ^= polynomial;
      }
      xbit >>= 1U;
    }
  }
  return crc;
}

bool IsFiniteCommand(const b2z1_real::B2LowCommand& command) {
  for (std::size_t index = 0; index < 12; ++index) {
    if (!std::isfinite(command.q[index]) || !std::isfinite(command.dq[index]) ||
        !std::isfinite(command.kp[index]) || !std::isfinite(command.kd[index]) ||
        !std::isfinite(command.tau[index])) {
      return false;
    }
  }
  return true;
}

}  // namespace

class B2Sdk2Bridge {
 public:
  B2Sdk2Bridge() : private_node_("~") {
    private_node_.param<std::string>("b2_bridge/network_interface", network_interface_, "eth0");
    private_node_.param("b2_bridge/command_rate", command_rate_, 500.0);
    private_node_.param("b2_bridge/command_timeout", command_timeout_, 0.10);
    private_node_.param("b2_bridge/release_motion_service", release_motion_service_, false);
    private_node_.param("b2_bridge/damping_kd", damping_kd_, 8.0);
    private_node_.param<std::string>("topics/b2_state", state_topic_, "/b2/low_state");
    private_node_.param<std::string>("topics/b2_command", command_topic_, "/b2/low_command");

    if (network_interface_.empty()) {
      throw std::runtime_error("B2 network interface parameter is empty");
    }
    unitree::robot::ChannelFactory::Instance()->Init(0, network_interface_);
    InitializeLowCommand();

    state_publisher_ = node_.advertise<b2z1_real::B2LowState>(state_topic_, 1);
    command_subscriber_ = node_.subscribe(command_topic_, 1, &B2Sdk2Bridge::CommandCallback, this);
    low_command_publisher_.reset(
        new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(kLowCommandTopic));
    low_command_publisher_->InitChannel();
    low_state_subscriber_.reset(
        new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(kLowStateTopic));
    low_state_subscriber_->InitChannel(
        std::bind(&B2Sdk2Bridge::LowStateCallback, this, std::placeholders::_1), 1);

    if (release_motion_service_) {
      ReleaseMotionService();
    } else {
      ROS_WARN("B2 sport service release is disabled. Low-level commands will only work after the "
               "operator releases the active motion service.");
    }
    write_timer_ = node_.createWallTimer(ros::WallDuration(1.0 / command_rate_),
                                         &B2Sdk2Bridge::WriteCallback, this);
    ROS_INFO_STREAM("B2 SDK2 bridge ready on " << network_interface_ << "; no DDS command is sent "
                                                << "until an enabled ROS command is received.");
  }

 private:
  void InitializeLowCommand() {
    low_command_.head()[0] = 0xFE;
    low_command_.head()[1] = 0xEF;
    low_command_.level_flag() = 0xFF;
    low_command_.gpio() = 0;
    for (std::size_t index = 0; index < 20; ++index) {
      auto& motor = low_command_.motor_cmd()[index];
      motor.mode() = 0x0A;
      motor.q() = kPositionStop;
      motor.kp() = 0.0F;
      motor.dq() = kVelocityStop;
      motor.kd() = 0.0F;
      motor.tau() = 0.0F;
    }
  }

  void ReleaseMotionService() {
    motion_switcher_.SetTimeout(5.0F);
    motion_switcher_.Init();
    for (int attempt = 0; attempt < 20; ++attempt) {
      std::string robot_form;
      std::string motion_name;
      const int32_t check_result = motion_switcher_.CheckMode(robot_form, motion_name);
      if (check_result != 0) {
        throw std::runtime_error("B2 CheckMode failed with code " + std::to_string(check_result));
      }
      if (motion_name.empty()) {
        return;
      }
      if (attempt == 0) {
        ROS_WARN_STREAM("Releasing active B2 motion service '" << motion_name
                                                               << "' by operator request");
        const int32_t release_result = motion_switcher_.ReleaseMode();
        if (release_result != 0) {
          throw std::runtime_error("B2 ReleaseMode failed with code " +
                                   std::to_string(release_result));
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    throw std::runtime_error("B2 motion service did not deactivate within 5 seconds");
  }

  void CommandCallback(const b2z1_real::B2LowCommand::ConstPtr& message) {
    if (!IsFiniteCommand(*message)) {
      ROS_ERROR_THROTTLE(1.0, "Rejected non-finite B2 command");
      return;
    }
    std::lock_guard<std::mutex> lock(command_mutex_);
    latest_command_ = *message;
    command_received_at_ = ros::WallTime::now();
    have_command_ = true;
    if (message->enabled) {
      activated_ = true;
    }
  }

  void LowStateCallback(const void* raw_message) {
    auto state = *static_cast<const unitree_go::msg::dds_::LowState_*>(raw_message);
    const bool crc_ok = state.crc() ==
        Crc32Core(reinterpret_cast<uint32_t*>(&state),
                  (sizeof(unitree_go::msg::dds_::LowState_) >> 2U) - 1U);

    b2z1_real::B2LowState output;
    output.header.stamp = ros::Time::now();
    output.header.frame_id = "base_link";
    const auto& imu = state.imu_state();
    // Unitree quaternion storage is [w, x, y, z]; geometry_msgs is [x, y, z, w].
    output.orientation.w = imu.quaternion()[0];
    output.orientation.x = imu.quaternion()[1];
    output.orientation.y = imu.quaternion()[2];
    output.orientation.z = imu.quaternion()[3];
    output.angular_velocity.x = imu.gyroscope()[0];
    output.angular_velocity.y = imu.gyroscope()[1];
    output.angular_velocity.z = imu.gyroscope()[2];
    output.linear_acceleration.x = imu.accelerometer()[0];
    output.linear_acceleration.y = imu.accelerometer()[1];
    output.linear_acceleration.z = imu.accelerometer()[2];
    for (std::size_t index = 0; index < 3; ++index) {
      output.rpy[index] = imu.rpy()[index];
    }
    bool finite = true;
    for (std::size_t index = 0; index < 12; ++index) {
      const auto& motor = state.motor_state()[index];
      output.q[index] = motor.q();
      output.dq[index] = motor.dq();
      output.tau_est[index] = motor.tau_est();
      output.motor_temperature[index] = motor.temperature();
      finite = finite && std::isfinite(motor.q()) && std::isfinite(motor.dq());
    }
    for (std::size_t index = 0; index < 4; ++index) {
      output.foot_force[index] = state.foot_force()[index];
    }
    output.tick = state.tick();
    output.crc_ok = crc_ok;
    output.valid = finite && state.head()[0] == 0xFE && state.head()[1] == 0xEF;
    if (!crc_ok) {
      ROS_ERROR_THROTTLE(1.0, "B2 LowState CRC check failed");
    }
    state_publisher_.publish(output);
  }

  void SetDampingCommand() {
    for (std::size_t index = 0; index < 12; ++index) {
      auto& motor = low_command_.motor_cmd()[index];
      motor.q() = kPositionStop;
      motor.kp() = 0.0F;
      motor.dq() = 0.0F;
      motor.kd() = static_cast<float>(damping_kd_);
      motor.tau() = 0.0F;
    }
  }

  void WriteCallback(const ros::WallTimerEvent&) {
    b2z1_real::B2LowCommand command;
    bool fresh_enabled = false;
    bool activated = false;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      activated = activated_;
      if (have_command_) {
        command = latest_command_;
        fresh_enabled = command.enabled &&
            (ros::WallTime::now() - command_received_at_).toSec() <= command_timeout_;
      }
    }
    if (!activated) {
      return;
    }
    if (fresh_enabled) {
      for (std::size_t index = 0; index < 12; ++index) {
        auto& motor = low_command_.motor_cmd()[index];
        motor.q() = static_cast<float>(command.q[index]);
        motor.dq() = static_cast<float>(command.dq[index]);
        motor.kp() = static_cast<float>(std::max(0.0, command.kp[index]));
        motor.kd() = static_cast<float>(std::max(0.0, command.kd[index]));
        motor.tau() = static_cast<float>(command.tau[index]);
      }
    } else {
      SetDampingCommand();
      ROS_ERROR_THROTTLE(1.0, "B2 command disabled/stale; sending damping command");
    }
    low_command_.crc() = Crc32Core(reinterpret_cast<uint32_t*>(&low_command_),
                                   (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2U) - 1U);
    low_command_publisher_->Write(low_command_);
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  ros::Publisher state_publisher_;
  ros::Subscriber command_subscriber_;
  ros::WallTimer write_timer_;
  std::string network_interface_;
  std::string state_topic_;
  std::string command_topic_;
  double command_rate_{500.0};
  double command_timeout_{0.10};
  double damping_kd_{8.0};
  bool release_motion_service_{false};
  bool activated_{false};
  bool have_command_{false};
  std::mutex command_mutex_;
  ros::WallTime command_received_at_;
  b2z1_real::B2LowCommand latest_command_;
  unitree_go::msg::dds_::LowCmd_ low_command_{};
  unitree::robot::b2::MotionSwitcherClient motion_switcher_;
  unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> low_command_publisher_;
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> low_state_subscriber_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "b2_sdk2_bridge");
  try {
    B2Sdk2Bridge bridge;
    ros::AsyncSpinner spinner(2);
    spinner.start();
    ros::waitForShutdown();
  } catch (const std::exception& error) {
    ROS_FATAL_STREAM("B2 SDK2 bridge failed: " << error.what());
    return 1;
  }
  return 0;
}
