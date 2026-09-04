#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <ros/ros.h>

#include <b2z1_real/Z1Command.h>
#include <b2z1_real/Z1State.h>
#include <unitree_arm_sdk/control/unitreeArm.h>

namespace {

bool IsFiniteCommand(const b2z1_real::Z1Command& command) {
  for (std::size_t index = 0; index < 6; ++index) {
    if (!std::isfinite(command.q[index]) || !std::isfinite(command.dq[index])) {
      return false;
    }
  }
  return true;
}

}  // namespace

class Z1SdkBridge {
 public:
  Z1SdkBridge() : private_node_("~") {
    private_node_.param("z1_bridge/has_gripper", has_gripper_, true);
    private_node_.param("z1_bridge/state_rate", state_rate_, 250.0);
    private_node_.param("z1_bridge/command_timeout", command_timeout_, 0.10);
    private_node_.param<std::string>("topics/z1_state", state_topic_, "/z1/low_state");
    private_node_.param<std::string>("topics/z1_command", command_topic_, "/z1/low_command");

    arm_.reset(new UNITREE_ARM::unitreeArm(has_gripper_));
    arm_->sendRecvThread->start();
    state_publisher_ = node_.advertise<b2z1_real::Z1State>(state_topic_, 1);
    command_subscriber_ = node_.subscribe(command_topic_, 1, &Z1SdkBridge::CommandCallback, this);
    control_timer_ = node_.createWallTimer(ros::WallDuration(0.002), &Z1SdkBridge::ControlCallback, this);
    state_timer_ = node_.createWallTimer(ros::WallDuration(1.0 / state_rate_),
                                         &Z1SdkBridge::PublishState, this);
    ROS_INFO("Z1 SDK bridge connected. The arm remains in its current FSM until an enabled command arrives.");
  }

  ~Z1SdkBridge() {
    if (arm_) {
      if (active_) {
        arm_->setFsm(UNITREE_ARM::ArmFSMState::PASSIVE);
      }
      arm_->sendRecvThread->shutdown();
    }
  }

 private:
  void CommandCallback(const b2z1_real::Z1Command::ConstPtr& message) {
    if (!IsFiniteCommand(*message)) {
      ROS_ERROR_THROTTLE(1.0, "Rejected non-finite Z1 command");
      return;
    }
    std::lock_guard<std::mutex> lock(command_mutex_);
    latest_command_ = *message;
    command_received_at_ = ros::WallTime::now();
    have_command_ = true;
  }

  void ControlCallback(const ros::WallTimerEvent&) {
    b2z1_real::Z1Command command;
    bool fresh_enabled = false;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      if (have_command_) {
        command = latest_command_;
        fresh_enabled = command.enabled &&
            (ros::WallTime::now() - command_received_at_).toSec() <= command_timeout_;
      }
    }

    if (!fresh_enabled) {
      if (active_) {
        ROS_ERROR("Z1 command disabled/stale; switching the arm to PASSIVE");
        arm_->setFsm(UNITREE_ARM::ArmFSMState::PASSIVE);
        active_ = false;
      }
      return;
    }

    if (!active_) {
      ROS_WARN("Enabling Z1 JOINTCTRL at the current joint position");
      arm_->startTrack(UNITREE_ARM::ArmFSMState::JOINTCTRL);
      active_ = true;
    }
    Vec6 q;
    Vec6 dq;
    for (std::size_t index = 0; index < 6; ++index) {
      q(static_cast<Eigen::Index>(index)) = command.q[index];
      dq(static_cast<Eigen::Index>(index)) = command.dq[index];
    }
    arm_->q = q;
    arm_->qd = dq;
    arm_->setArmCmd(q, dq);
  }

  void PublishState(const ros::WallTimerEvent&) {
    b2z1_real::Z1State output;
    output.header.stamp = ros::Time::now();
    output.header.frame_id = "z1_mount";
    output.joints.header = output.header;
    const Vec6 q = arm_->lowstate->getQ();
    const Vec6 dq = arm_->lowstate->getQd();
    const Vec6 tau = arm_->lowstate->getTau();
    output.joints.name.reserve(6);
    output.joints.position.reserve(6);
    output.joints.velocity.reserve(6);
    output.joints.effort.reserve(6);
    for (std::size_t index = 0; index < 6; ++index) {
      output.joints.name.push_back("joint" + std::to_string(index + 1));
      output.joints.position.push_back(q(static_cast<Eigen::Index>(index)));
      output.joints.velocity.push_back(dq(static_cast<Eigen::Index>(index)));
      output.joints.effort.push_back(tau(static_cast<Eigen::Index>(index)));
    }

    bool valid = true;
    const std::size_t count = std::min(
        {std::size_t{7}, arm_->lowstate->temperature.size(), arm_->lowstate->errorstate.size(),
         arm_->lowstate->isMotorConnected.size()});
    for (std::size_t index = 0; index < count; ++index) {
      output.temperature[index] = static_cast<uint8_t>(
          std::max(0, std::min(255, arm_->lowstate->temperature[index])));
      output.error[index] = arm_->lowstate->errorstate[index];
      output.connection[index] = arm_->lowstate->isMotorConnected[index];
      valid = valid && ((output.error[index] & static_cast<uint8_t>(~0x40U)) == 0U) &&
          output.connection[index] == 0U;
    }
    for (Eigen::Index index = 0; index < 6; ++index) {
      valid = valid && std::isfinite(q(index)) && std::isfinite(dq(index));
    }
    output.valid = valid;
    state_publisher_.publish(output);
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  ros::Publisher state_publisher_;
  ros::Subscriber command_subscriber_;
  ros::WallTimer control_timer_;
  ros::WallTimer state_timer_;
  std::unique_ptr<UNITREE_ARM::unitreeArm> arm_;
  std::string state_topic_;
  std::string command_topic_;
  bool has_gripper_{true};
  bool active_{false};
  bool have_command_{false};
  double state_rate_{250.0};
  double command_timeout_{0.10};
  std::mutex command_mutex_;
  ros::WallTime command_received_at_;
  b2z1_real::Z1Command latest_command_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "z1_sdk_bridge");
  try {
    Z1SdkBridge bridge;
    ros::AsyncSpinner spinner(2);
    spinner.start();
    ros::waitForShutdown();
  } catch (const std::exception& error) {
    ROS_FATAL_STREAM("Z1 SDK bridge failed: " << error.what());
    return 1;
  }
  return 0;
}
