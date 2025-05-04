# Food Finder - Image Classification for Food Recognition

## Project Overview
Food Finder is a web-based application that uses machine learning to recognize food items from user-uploaded images. It consists of a simple frontend, a Flask backend, and a placeholder for your ML model.

---

## Folder Structure
```
backend/         # Flask API and model code
frontend/        # Simple HTML/JS frontend
```

---

## Local Setup

### 1. Clone the repository
```
git clone https://github.com/kalyanganala28/food-finder.git
cd food-finder
```

### 2. Backend Setup
```
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### 3. Frontend Setup
Open `frontend/index.html` in your browser. (You may need to allow CORS or run a simple HTTP server.)

---

## Deploying on AWS EC2

### 1. Launch an EC2 Instance
- Use Ubuntu 24.04 LTS (or 20.04) for free tier eligibility.
- Allow inbound ports 22 (SSH), 5000 (Flask), and 80 (HTTP) in your security group.

### 2. Connect to Your Instance (Open a Terminal)
- **What is a terminal?**
  - On Windows: Use PowerShell, Command Prompt, or Git Bash. For SSH, Git Bash is recommended.
  - On Mac: Open the "Terminal" app from Applications > Utilities.
  - On Linux: Use your default terminal app.
- **SSH Command:**
  ```sh
  ssh -i path/to/your-key.pem ubuntu@<your-ec2-public-dns>
  ```
  - Replace `path/to/your-key.pem` with the path to your downloaded key pair file.
  - Replace `<your-ec2-public-dns>` with your instance's public DNS (find it in the AWS EC2 console).

### 3. Install Python and Git
- **Why?** These are needed to run your app and get your code from GitHub.
- **Command:**
  ```sh
  sudo apt update
  sudo apt install python3-pip git
  ```
  - `sudo apt update` updates the package list.
  - `sudo apt install python3-pip git` installs Python, pip, and git.

### 4. Get Your Code onto the Server
- **Why?** You need to copy your project from GitHub to your EC2 instance.
- **Command:**
  ```sh
  git clone https://github.com/kalyanganala28/food-finder.git
  cd food-finder/backend
  ```
  - `git clone ...` downloads your code.
  - `cd food-finder/backend` moves into the backend folder.

### 5. Install Python Packages
- **Why?** These are the libraries your app needs to run.
- **Command:**
  ```sh
  pip3 install -r requirements.txt
  ```
  - This reads `requirements.txt` and installs Flask, Pillow, etc.

### 6. Run the Backend
```
python3 app.py
```
- The API will be available at `http://<your-ec2-public-dns>:5000/predict`

### 7. Serve the Frontend
- You can copy `frontend/index.html` to your local machine and access it, or use a simple web server (e.g., Python's `http.server`).

---

## Model Integration
- Replace the code in `backend/model.py` with your trained model loading and prediction logic.
- Add any extra dependencies to `backend/requirements.txt`.

---

## Security & Scalability
- For production, use `gunicorn` and `nginx`.
- Store images in AWS S3 for scalability.
- Use HTTPS for secure communication.

---

## Testing
- Test locally with different food images.
- Test on AWS EC2 with real uploads.

---

## Contributing
- Fork the repo, make changes, and submit a pull request.

---

## License
MIT 