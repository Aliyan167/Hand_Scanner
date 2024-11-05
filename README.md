# Hand_scanner
Features
Hand Length Measurement: Calculates the full length of the hand from the base of the palm to the tip of the middle finger.
Trigger Distance: Measures the distance from the base of the hand to a selected point near the finger trigger area, which can be used in designing equipment like firearms or sports equipment.
Grip Length: Measures the span from the base of the palm to a specified grip point on the fingers, useful in sports or ergonomic design.

<h1>How it works </h1>
  
Input Image: The user provides an image of their hand, ideally placed alongside a reference object for accurate scaling.
Image Processing: The program processes the image to identify key landmarks on the hand.
Measurement Calculation: Using the detected landmarks and the reference object, the program calculates:
Hand length
Trigger distance
Grip length


<h1>Output:</h1>
The measurements are displayed, allowing users to record or apply the values for various applications. 

<h1>Requirements</h1>

Python 3.x
OpenCV for image processing
