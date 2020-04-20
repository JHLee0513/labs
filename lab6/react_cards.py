#!/usr/bin/env python3

import asyncio
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, '../lab5')
import imgclassification as clf

import cozmo

try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')


# Define a decorator as a subclass of Annotator; displays battery voltage
class BatteryAnnotator(cozmo.annotate.Annotator):
    def apply(self, image, scale):
        d = ImageDraw.Draw(image)
        bounds = (0, 0, image.width, image.height)
        batt = self.world.robot.battery_voltage
        text = cozmo.annotate.ImageText('BATT %.1fv' % batt, color='green')
        text.render(d, bounds)

async def run(robot: cozmo.robot.Robot):
    '''The run method runs once the Cozmo SDK is connected.'''

    #add annotators for battery level and ball bounding box
    robot.world.image_annotator.add_annotator('battery', BatteryAnnotator)

    robot.move_lift(-4)
    camera = robot.camera
    camera.enable_auto_exposure()
    img_clf = clf.ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('../lab5/train/')
    train_data = img_clf.extract_image_features(train_raw)
    img_clf.train_classifier(train_data, train_labels)

    # Array to get predictions over a sequence of frames
    class_count = np.zeros(4) # None, order, drone, hands
    frame_count = 0

    try:
        while True:
            #get camera image
            event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)                

            #convert camera image to opencv format
            opencv_image = np.asarray(event.image)

            #make predictions
            test_data = img_clf.extract_image_features([opencv_image])
            predicted_labels = img_clf.predict_labels(test_data)
            #record predicted labels
            if predicted_labels == 'order':
                class_count[1] += 1
            elif predicted_labels == 'drone':
                class_count[2] += 1
            elif predicted_labels == 'inspection':
                class_count[3] += 1
            else:
                class_count[0] += 1

            #behavior handling for found non-null class
            if frame_count % 5 == 0:
                max_idx = np.argmax(class_count)
                if max_idx == 1:
                    #order
                    await robot.say_text("order detected").wait_for_completed()
                    #rotate with lift up
                    robot.move_lift(2)
                    time.sleep(.2)
                    await robot.turn_in_place(cozmo.util.degrees(270)).wait_for_completed()
                    await robot.turn_in_place(cozmo.util.degrees(-180)).wait_for_completed()
                    robot.move_lift(-2)
                    time.sleep(.2)
                elif max_idx == 2:
                    #drone
                    await robot.say_text("drone detected").wait_for_completed()
                    # Sing notes from tutorial 01_11
                    notes = [
                        cozmo.song.SongNote(cozmo.song.NoteTypes.C3, cozmo.song.NoteDurations.Half),
                        cozmo.song.SongNote(cozmo.song.NoteTypes.C3, cozmo.song.NoteDurations.ThreeQuarter),
                        cozmo.song.SongNote(cozmo.song.NoteTypes.Rest, cozmo.song.NoteDurations.Quarter),
                        cozmo.song.SongNote(cozmo.song.NoteTypes.C3, cozmo.song.NoteDurations.Quarter),
                        cozmo.song.SongNote(cozmo.song.NoteTypes.C3, cozmo.song.NoteDurations.Whole) ]
                    await robot.play_song(notes, loop_count=1).wait_for_completed()
                elif max_idx == 3:
                    #inspection
                    await robot.say_text("inspection detected").wait_for_completed()
                    #play animation experimented in tutorial 01_08
                    await robot.play_anim_trigger(cozmo.anim.Triggers.CubePounceLoseSession, ignore_body_track=True).wait_for_completed()
                    await robot.turn_in_place(cozmo.util.degrees(25)).wait_for_completed()
                else:
                    # Keep searching for objects
                    await robot.turn_in_place(cozmo.util.degrees(10)).wait_for_completed()

                class_count = np.zeros(4)

    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)

if __name__ == '__main__':
    cozmo.run_program(run, use_viewer = True, force_viewer_on_top = True)

