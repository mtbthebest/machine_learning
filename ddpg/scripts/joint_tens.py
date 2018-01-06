#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np
import tensorflow as tf
import keras

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
        position= tf.placeholder(tf.float32, (15,))
        Joint.velocity =[]
        Joint.effort = []
        episode_start = rospy.Time.now()
        with tf.Session() as sess:
            
            while (rospy.Time.now() - episode_start < rospy.Duration(10)):    

                 while not rospy.is_shutdown():
                    step_start = rospy.Time.now()
                    target = np.random.uniform(0.,0.5,size=15)
                    Joint.position = sess.run(position, feed_dict={position:target})
                    print Joint.position
                    while (rospy.Time.now() - step_start < rospy.Duration(3)):    

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

# c = tf.placeholder(tf.float32, (15,))

# d =c 
# with tf.Session() as sess:
#       #a= tf.random_uniform((3,), 0,1)
     
#       for i in range(10):
#         f = sess.run(c, feed_dict={c: np.random.uniform(0,0.5,15)})
#         print f
