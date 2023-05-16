import cv2, time
import threading
# openCV's default VideoCapture cannot drop frames so if the CPU is overloaded the stream will tart to lag behind realtime.
# This class creates a treaded capture implementation that can stay up to date wit the stream and decodes frames only on demand
class ThreadedCapture:
    def __init__(self, name, FPS=2):
        # FPS = 1/X 
        # X = desired FPS
        self.FPS = 1/FPS
        self.FPS_MS = int(self.FPS * 1000)

        self.cap = cv2.VideoCapture(name)
        if not self.cap.isOpened():
            self.cap.release()
            raise Exception(f"Could not open stream ({name})")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Limit buffering to 1 frames
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def __del__(self):
        self.cap.release()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            with self.lock:
                ret = self.cap.grab()
                time.sleep(1/30) # Limit acquisition speed
            if not ret:
                break

    # retrieve latest frame
    def read(self):
        with self.lock:
            _, frame = self.cap.retrieve()
        return _, frame

if __name__ == '__main__':
    stream_url = 'https://s78.ipcamlive.com/streams/4etgocfj23fhnhsne/stream.m3u8'
    threaded_camera = ThreadedCapture(stream_url,2)
    while True:
        try:
            _, frame = threaded_camera.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(threaded_camera.FPS_MS)
        except AttributeError:
            pass
