#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np
import tensorflow as tf


class Joints:
    def __init__(self):
         
        rospy.init_node('joint_publisher')
        rate =12000
        r = rospy.Rate(rate)

        pub = rospy.Publisher('joint_states', JointState, queue_size=15)
        
        Joint = JointState()
        Joint.header = Header()
        Joint.header.stamp = rospy.Time.now()
        Joint.name = ['r_wheel_joint', 'l_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint', 'r_gripper_finger_joint', 'l_gripper_finger_joint', 'bellows_joint']
        Joint.position = np.array([])
        Joint.velocity =[]
        Joint.effort = []
        
        while not rospy.is_shutdown():
            Joint.position = np.append([], np.random.uniform(0.,0.5,size=len(Joint.name)))
            start = rospy.Time.now()

            while (rospy.Time.now() - start < rospy.Duration(3)):
                for i in {0,1,2,3,4,12,13,14}:
                    Joint.position[i] = 0
                Joint.header.stamp = rospy.Time.now()

                pub.publish(Joint)
                
                r.sleep()
                
            
            
if __name__ == '__main__':
    
    try:
        Joints()
    except rospy.ROSInterruptException:
        rospy.loginfo('Over')
    