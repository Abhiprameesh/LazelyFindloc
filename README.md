# LazelyFindLoc - Landmark Recognition System

A deep learning-based application that identifies famous landmarks from images. The system uses a pre-trained ResNet18 model fine-tuned on landmark datasets.

## Features

- Identify famous landmarks from uploaded images
- Web interface built with Streamlit
- Supports multiple landmark categories including:
  - Big Ben
  - Burj Khalifa
  - Eiffel Tower
  - Statue of Liberty
  - Taj Mahal

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LazelyFindLoc.git
   cd LazelyFindLoc
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app/app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload an image of a landmark using the file uploader

4. The app will predict and display the most likely landmark in the image

## Project Structure

```
LazelyFindLoc/
├── streamlit_app/      # Web application code
│   ├── app.py         # Main Streamlit application
│   └── util/          # Utility functions
│       └── inference.py
├── model/             # Pre-trained models (not included in git)
├── dataset/           # Sample dataset (not included in git)
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
