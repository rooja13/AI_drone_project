# code can be found from this website: https://gazebosim.org/api/transport/13/python.html
# this is the "Subscriber" code and the webpage provides a code walk through

from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node
import cv2
import numpy
import time

# The image_cb() callback function will:
# 1. Convert raw bytes to numpy array
# 2. Convert RGB to BGR
# 3. Run frame through neural network
# 4. Publish velocity command to /drone/cmd_vel

def image_cb(msg: Image):
	image_array = numpy.frombuffer(msg.data, numpy.uint8)
	image_array = numpy.reshape(image_array, (msg.height, msg.width, 3))
	image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
	
	# uncomment this to have a window pop up with the image
	# cv2.imshow("camera frame", image_bgr)
	# cv2.waitKey(1)
	
	print("Received Image: [{} x {}]".format(msg.height, msg.width)

def main():
	node = Node()
	topic_image_msg = "/drone/camera"
	
	# subscribe to a topic by registering a callback
	if node.subscribe(Image, topic_image_msg, image_cb):
		print("Subscribing to type {} on topic [{}]".format(
		Image, topic_image_msg))
	else:
		print("Error subscribing to topic [{}]".format(topic_image_msg))
		return
	
	# wait for shutdown
	try:
		while True:
			time.sleep(0.001)
	except KeyboardInterrupt:
		pass
	print("Done")

	
if __name__ == '__main__':
    main()
