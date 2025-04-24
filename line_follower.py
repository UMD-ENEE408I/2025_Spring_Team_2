#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import numpy as np
from std_msgs.msg import String

LINEAR_SPEED = 0.025
KP = 1.5/1000

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.subscription = rospy.Subscriber('video_topic/compressed', 
                                             CompressedImage,
                                             self.listener_callback)
        self.subscription
        self.publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.state = {
            "IDLE" : True,
            "FOLLOW": False,
            "LEFT": False,
            "RIGHT": False,
            "TJUNCT": False
        }
        

    def listener_callback(self, msg):
        # set up twist
        cmd = Twist()
        cmd.linear.x = 0
        cmd.angular.z = 0

        # process image
        current_frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        # hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        # lower = (0, 0, 200)
        # upper = (180, 30, 255)
        lower = (200, 200, 200)
        upper = (255, 255, 255)
        mask = cv2.inRange(current_frame, lower, upper)
        error = 0
        seg_image = cv2.bitwise_and(current_frame, current_frame, mask=mask)
        _, width, _ = seg_image.shape
        # detect line
        line = self.get_contour_data(mask)
        if self.state["IDLE"]:
            if line:
                self.switch_case("FOLLOW")
                rospy.loginfo("Idling")

        elif self.state["FOLLOW"]:
            if line:

                cv2.circle(seg_image, (line['x'], line['y']), 5, (0, 0, 255), 7)

                fitline = line['fitline']
                t0 = (0 - fitline[3])/fitline[1]
                t1 = (seg_image.shape[0]-fitline[3])/fitline[1]

                p0 = (fitline[2:4] + (t0 * fitline[0:2])).astype(np.uint32)
                p1 = (fitline[2:4] + (t1 * fitline[0:2])).astype(np.uint32)

                cv2.line(seg_image, tuple(p0.ravel()), tuple(p1.ravel()), (0, 255, 0), 2)

                if abs(line['angle']) < 60:
                    if line['angle'] > 0:
                        self.switch_case("LEFT")
                    else:
                        self.switch_case("RIGHT")

                else:
                    x = line['x']
                    error = x - width//2
                    cmd.linear.x = LINEAR_SPEED
                    cmd.angular.z = (float(error) * - KP)
                    rospy.loginfo("Following, error: {} | line angle: {} | angular z: {}".format(error, line['angle'], cmd.angular.z))
            else:
                self.switch_case("IDLE")
            # need some logic in here to detect the T
            # Thinking break the window up so that we can see only the top right and top left corners
            # if there is white in both of those spots at the same time we hit the T
            # and need to switch to the T case
        elif self.state["LEFT"]:
            if line:
                rospy.loginfo("Turning left, line angle: {}".format(line['angle']))
                if abs(line['angle']) < 60:
                        if line['y'] < 160: 
                            cmd.angular.z = 0.07
                            cmd.linear.x = LINEAR_SPEED
                        elif line['y'] > 320:
                            cmd.angular.z = 0.16
                            cmd.linear.x = LINEAR_SPEED
                        else:
                            cmd.angular.z = 0.11
                            cmd.linear.x = LINEAR_SPEED
                else:
                    self.switch_case("FOLLOW")
            else:
                self.switch_case("IDLE")
        elif self.state["RIGHT"]:
            if line:
                rospy.loginfo("Turning right, line angle: {}".format(line['angle']))
                if abs(line['angle']) < 60:
                    if line['y'] < 160: 
                        cmd.angular.z = -0.07
                        cmd.linear.x = LINEAR_SPEED
                        rospy.loginfo("running")

                    elif line['y'] > 320:
                        cmd.angular.z = -0.16
                        cmd.linear.x = LINEAR_SPEED
                    else:
                        cmd.angular.z = -0.11
                        cmd.linear.x = LINEAR_SPEED
                else:
                    self.switch_case("FOLLOW")
            else:
                self.switch_case("IDLE")
        elif self.state["TJUNCT"]:
            # create subscriber for audio message and wait for the thang
            speech_holder = ""
            cb = lambda msg: speech_holder = msg.data # dont even know if this works
            sub = rospy.Subscriber('/recognized_speech', String, cb)
            while (speech_holder.capitalize() != "LEFT" or speech_holder.capitalize() != "RIGHT"): # wait for speech recog, this is a bit janky
                pass
            self.switch_case(speech_holder) # also janky
            sub.unregister() # delete that thang

        self.publisher.publish(cmd)
        cv2.imshow("Camera Feed", seg_image)
        cv2.waitKey(1)

    def get_contour_data(self, mask):
        MIN_AREA_TRACK = 50
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        line = {}

        for contour in contours:
            M = cv2.moments(contour)

            if(M['m00'] > MIN_AREA_TRACK):
                line['x'] = int(M["m10"]/M["m00"])
                line['y'] = int(M["m01"]/M["m00"])
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                line['angle'] = 180*np.arctan2(vy,vx)/np.pi
                line['fitline'] = [vx, vy, x, y]

        return line

    def switch_case(self, case):
        for state in self.state:
            if state == case:
                self.state[case] = True
            else:
                self.state[state] = False


if __name__ == '__main__':
    node = ImageSubscriber()
    rospy.spin()