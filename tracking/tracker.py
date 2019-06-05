from tracking.utils import doNothing, getRandomColor, rectBoxToTrackBox


class Tracker:
    def __init__(self, life, trackBox):
        self.life = life
        self.activeTime = 0

        self.trackBox = trackBox
        self.paired = False

        self.color = getRandomColor()
        self.counted = False


class MultiTracker:
    def __init__(self, trackerLife, trackerActiveTime):
        self.trackers = []
        self.trackerLife = trackerLife
        self.trackerActiveTime = trackerActiveTime

    def add(self, tracker):
        self.trackers.append(tracker)

    def matchDetected(self, detectedBoxes, pairingFunction, onJustCounted=doNothing, onCounted=doNothing,
                      onNotCounted=doNothing):

        # Find best tracker for each detection box
        notPairedBoxes = pairingFunction(self.trackers, detectedBoxes)

        for box in notPairedBoxes or []:
            newTracker = Tracker(self.trackerLife, rectBoxToTrackBox(box))
            self.add(newTracker)

        # Update trackers' life
        self._updateTrackersLife(onJustCounted, onCounted, onNotCounted)
        self.resetPaired()

    def _updateTrackersLife(self, onJustCounted, onCounted, onNotCounted):
        for tracker in self.trackers:
            if tracker.paired:
                tracker.life = self.trackerLife

                if tracker.activeTime > self.trackerActiveTime:
                    if not tracker.counted:
                        onJustCounted(tracker)
                        tracker.counted = True
                    else:
                        onCounted(tracker)
                else:
                    onNotCounted(tracker)
            else:
                tracker.life -= 1
                if tracker.life == 0:
                    self.trackers.remove(tracker)

    def resetPaired(self):
        for t in self.trackers:
            t.paired = False
