import roslib; roslib.load_manifest('borg_pioneer')
import rospy
import logging
import util.loggingextra as loggingextra

from util.marytts import MaryTTS
from std_msgs.msg import String

def received_speech(data):
    TTS.say(data.data)

if __name__ == "__main__":
    logging.getLogger('Borg.Brain').addHandler(loggingextra.ScreenOutput())
    logging.getLogger('Borg.Brain').setLevel(logging.INFO)

    rospy.init_node("MaryTTS")

    TTS = MaryTTS()
    TTS.set_voice(rospy.get_param("voice", "Prudence"))
    TTS.set_host(
        rospy.get_param("host", "localhost"),
        rospy.get_param("port", "59125")
    )

    topic = rospy.get_param("topic", "/speech")
    pub = rospy.Subscriber(topic, String, received_speech)
    rospy.spin()
