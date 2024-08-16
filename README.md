# Car Damage Assessment System


## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

## Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd car-damage-assessment
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`

## Using the System

1. On the web interface, click the "Upload Image" button to select an image of a damaged car.

2. The system will process the image and display the following:
   - Original image
   - Annotated damage image
   - Part segmentation heatmap
   - Damage segmentation heatmap
   - List of predicted parts to replace with probabilities
   - Top 5 parts with highest probabilities

3. Analyze the results to assess the damage and determine necessary repairs.

## File Structure

- `app.py`: Main Flask application that handles the web interface and image processing
- `AI.py`: Contains functions for image processing and part prediction (imported by `app.py`)
- `requirements.txt`: List of Python packages required for the project
- `improved_car_damage_prediction_model.h5`: Pre-trained deep learning model for part replacement prediction (included in the repository)
- `cars117.json`: JSON file containing the list of car parts (included in the repository)
- `templates/index.html`: HTML template for the web interface
- `static/`: Directory for storing static files (CSS, JavaScript, images)
- `uploads/`: Directory for temporarily storing uploaded images



## Future Enhancements

- Integration with a cost estimation module for repair pricing
- Expansion of the model to cover a wider range of vehicle types and damage scenarios
- Development of a more interactive user interface for detailed damage exploration
- Implementation of a feedback loop to continuously improve model accuracy based on real-world repair data
