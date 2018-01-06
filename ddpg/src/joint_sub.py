#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


def sub(msg):
    rospy.loginfo (msg.position)

if __name__ == '__main__':
    
    rospy.init_node('joint_subscriber')
        

    pub = rospy.Subscriber('joint_states', JointState, callback=sub)
        
    rospy.spin()