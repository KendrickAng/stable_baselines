/*
 * This node has 2 responsibilities:
 * 1. Interpret the sensor_msgs/Joy messages from the /joy topic
 * 2. Arbitrate between sending manual and autonomous commands to the actuators
 * 
 * Output: A message of type geometry_msgs/Twist, where
 * msg.linear.x is the desired speed of the vehicle
 * msg.linear.z is 1 if the brakes are activated, 0 otherwise
 * msg.angular.z is the desired steering angle of the vehicle
 * All other fields should be 0
 * 
 * Callbacks:
 * 1. joystickCallback: interprets the Joy messages and publishes manual and brake commands
 * 2. navCmdVelCallback: forwards the cmd_vel_nav message iff the vehicle is in autonomous mode
 */

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/Twist.h>

enum class NavMode
{
  Brake,
  Manual,
  Autonomous
};

class JoystickTeleop
{
public:
  JoystickTeleop();

private:
  ros::NodeHandle nh_;

  ros::Subscriber joystick_sub_;
  ros::Subscriber nav_cmd_vel_sub_;
  ros::Publisher cmd_vel_pub_;

  NavMode current_nav_mode_;

  geometry_msgs::Twist cmd_vel_manual_;
  geometry_msgs::Twist cmd_vel_autonomous_;

  double MAX_SPEED = 1.0;
  double MAX_STEERING_ANGLE = 1.0;

  void joystickCallback(const sensor_msgs::Joy::ConstPtr &joy_msg);
  void navCmdVelCallback(const geometry_msgs::Twist::ConstPtr &nav_cmd_vel_msg);
};

JoystickTeleop::JoystickTeleop()
{
  ros::NodeHandle private_nh("~");

  std::string joy_topic;
  std::string nav_cmd_vel_topic;
  std::string final_cmd_vel_topic;

  ROS_ASSERT(private_nh.getParam("joy_topic", joy_topic));
  ROS_ASSERT(private_nh.getParam("nav_cmd_vel_topic", nav_cmd_vel_topic));
  ROS_ASSERT(private_nh.getParam("final_cmd_vel_topic", final_cmd_vel_topic));

  joystick_sub_ = nh_.subscribe(joy_topic, 1, &JoystickTeleop::joystickCallback, this);
  nav_cmd_vel_sub_ = nh_.subscribe(nav_cmd_vel_topic, 1, &JoystickTeleop::navCmdVelCallback, this);
  cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(final_cmd_vel_topic, 1);

  current_nav_mode_ = NavMode::Brake;
}

void JoystickTeleop::joystickCallback(const sensor_msgs::Joy::ConstPtr &joy_msg)
{
  // Step 1: map the joy_msg fields to the physical buttons
  int buttonA = joy_msg->buttons[0];
  int buttonB = joy_msg->buttons[1];
  int buttonX = joy_msg->buttons[2];
  double forward_axes = joy_msg->axes[4];
  double steering_axes = joy_msg->axes[0];

  // Step 2: change the current_nav_mode_ as necessary
  if (buttonB == 1)
  {
    current_nav_mode_ = NavMode::Brake;
    ROS_INFO("Joystick Teleop: Brake Mode!");
  }
  else if (buttonX == 1)
  {
    current_nav_mode_ = NavMode::Manual;
    ROS_INFO("Joystick Teleop: Manual Mode!");
  }
  else if (buttonA == 1)
  {
    if (current_nav_mode_ == NavMode::Brake)
    {
      current_nav_mode_ = NavMode::Autonomous;
      ROS_INFO("Joystick Teleop: Autonomous Mode!");
    }
    else if (current_nav_mode_ == NavMode::Autonomous)
    {
      ROS_INFO("Joystick Teleop: Already in Autonomous Mode");
    }
    else
    {
      ROS_INFO("Joystick Teleop: Can only go to Autonomous Mode after Hard Brake!");
    }
  }
  else
  {
    // Empty Else
  }

  // Step 3: perform manual controls as necessary
  if (current_nav_mode_ == NavMode::Manual)
  {
    cmd_vel_manual_.linear.x = forward_axes * MAX_SPEED;
    cmd_vel_manual_.angular.z = steering_axes * MAX_STEERING_ANGLE;
    cmd_vel_manual_.linear.z = 0;
    cmd_vel_pub_.publish(cmd_vel_manual_);
  }
  else if (current_nav_mode_ == NavMode::Brake)
  {
    cmd_vel_manual_.linear.x = 0;
    cmd_vel_manual_.angular.z = steering_axes * MAX_STEERING_ANGLE;
    cmd_vel_manual_.linear.z = 1;
    cmd_vel_pub_.publish(cmd_vel_manual_);
  }
  else
  {
    // Empty Else
  }
}

void JoystickTeleop::navCmdVelCallback(const geometry_msgs::Twist::ConstPtr &nav_cmd_vel_msg)
{
  if (current_nav_mode_ == NavMode::Autonomous)
  {
    cmd_vel_autonomous_.linear.x = nav_cmd_vel_msg->linear.x;
    cmd_vel_autonomous_.angular.z = nav_cmd_vel_msg->angular.z;
    ROS_INFO("Joystick Teleop: Publishing Autonomous Commands");
    cmd_vel_pub_.publish(cmd_vel_autonomous_);
  }
  else
  {
    return;
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "joystick_teleop_node");
  JoystickTeleop joystick_teleop_obj;
  ros::spin();
  return 0;
}
