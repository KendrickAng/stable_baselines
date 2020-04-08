#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import RPi.GPIO as IO
from gpiozero import AngularServo

class ActuatorsController:
    def __init__(self):
        self._cmd_vel_sub = rospy.Subscriber("/cmd_vel_final", Twist, self.cmd_vel_callback)

        self._max_speed = 1.0
        self._max_steering = 1.0
        
        self._steering_offset = 5.0

        IO.setwarnings(False)
        IO.setmode(IO.BCM)
        IO.setup(12, IO.OUT)
        IO.setup(23, IO.OUT, initial=1)
        self._p = IO.PWM(12, 500)
        self._p.start(0)

        self._servo = AngularServo(13, min_angle=-90, max_angle=90)

    def cmd_vel_callback(self, msg):

        # Step 1: Control the speed
        if msg.linear.z != 0:
            duty_cycle = 0
            print("Actuators: Braking!")
        else:
            desired_speed = msg.linear.x
            duty_cycle = desired_speed / self._max_speed * 50

        print("Actuators: duty_cycle:", duty_cycle)
        if duty_cycle >= 0:
            self._p.ChangeDutyCycle(duty_cycle)

        # Step 2: Control the steering
        desired_steering_angle = msg.angular.z
        self._servo.angle = desired_steering_angle / self._max_steering * -85.0 - self._steering_offset
        print("Actuators: The desired steering angle is", desired_steering_angle)
   

if __name__ == '__main__':
    rospy.init_node('actuators_controller')
    ActuatorsController()
    rospy.spin()
