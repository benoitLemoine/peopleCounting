## 1. Datasets

The application was developped using 4 different datasets as benchmark.
These datasets are the following:

 - ChockePoint dataset (CP)
 - MIVIA dataset (MIVIA)
 - Multiple Object Tracking Benchmark dataset (MOT)
 - People Counting Dataset (PCDS)

Under **/datasets** you can find scripts for formatting the groundtruth of each one into a standard one as well as scripts to verify the groundtruth for a given video.
The standard groundtruth I use is a bit special because it only gives the appearance time of each person.
There are also 2 scripts to manually create formatted groundtruth file (for instance for PCDS, there are by default no frame by frame groundtruth).

## 2. Detection

To detect people, I use a CNN architecture called YOLOv3 (see credits). Because I wanted something flexible in case of I needed to use another CNN, I created a kind of "standard net class", so I adapted both YOLOv3 and Tiny YOLOv3 in this format.
The class for each one as well as the sources for both networks can be found under **/detection**.

## 3. Tracking

Under **/tracking**  can be found the source code for a few things such as:

 - pairing functions
 - trackers
 - result exporter (for the standard groundtruth format)

The file **computeVideosCounting .py** is a complete implementation of the whole counting algorithm.
What it does is iterating on every single video, perform counting and export the results with the standard groundtruth format.

As you can see, there is a bunch of parameters and I will try to explain it briefly:

 - **Tracker life:** it is the time a tracker stay alive  if it is not paired
 - **Tracker active time:** it is the time a tracker should stay paired to be counted
 - **Pairing function:** it is the way the tracker try to match detected boxes with trackers' boxes
 - **onJustCounted/onCounted/onNotCounted:** these are lambda functions used to execute a given code on a specific events (by default, this is how I export the result when a new person is counted)

## 4. Credits

This is a copy of the following repository : https://github.com/YunYang1994/tensorflow-yolov3
Thanks to YunYang1994 for sharing it !

I also use the tiny yolov3 CNN from the repository : https://github.com/zzh8829/yolov3-tf2


