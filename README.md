# FarmSafe
This is Group 9's final project work for Introduction to Artificial Intelligence

This project is a Flask application designed as part of the Intro to AI course final project. This README file will guide you on how to set up and host this application on a local server or on the cloud using Vercel.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Local Installation](#local-installation)
4. [Hosting on Vercel](#hosting-on-vercel)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Project Overview

This Flask application showcases AI functionalities learned throughout the course. It includes:

- [List major features and functionalities]

## Getting Started

### Prerequisites

- Python 3.x
- Pip (Python package installer)
- Git

## Local Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Run the application:**

    ```sh
    python app.py
    ```

5. **Open your browser and go to:**

    ```
    http://127.0.0.1:5000
    ```

## Hosting on Vercel

1. **Install the Vercel CLI:**

    Download and install the Vercel CLI from [here](https://vercel.com/download).

2. **Log in to Vercel:**

    ```sh
    vercel login
    ```

3. **Initialize your Vercel project:**

    Navigate to your project directory and initialize your Vercel project:

    ```sh
    vercel init
    ```

4. **Create a `vercel.json` configuration file:**

    In the root directory of your project, create a file named `vercel.json` with the following content:

    ```json
    {
      "version": 2,
      "builds": [
        {
          "src": "app.py",
          "use": "@vercel/python"
        }
      ],
      "routes": [
        {
          "src": "/(.*)",
          "dest": "app.py"
        }
      ]
    }
    ```

5. **Deploy your application:**

    Deploy your application to Vercel:

    ```sh
    vercel
    ```

6. **Open your application:**

    Once deployed, Vercel will provide you with a URL where your application is hosted. Open this URL in your browser to view your application.

## Usage

1. **Access the application locally:**

    Open your browser and navigate to `http://127.0.0.1:5000`.

2. **Access the application on Vercel:**

    Open your browser and navigate to the URL provided by Vercel.

## Contributing

Feel free to fork this repository and submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

