# How to Run the Bitcoin Price Prediction Project

This document provides detailed instructions on how to set up and run the Bitcoin Price Prediction project.

## Prerequisites

- Python 3.8+ (for backend)
- Node.js 14+ (for frontend)
- npm (included with Node.js)

## Initial Setup

### 1. Backend Setup

1. **Create and activate a virtual environment:**

   Windows:
   ```
   cd backend
   python -m venv venv
   venv\Scripts\activate
   ```

   Linux/Mac:
   ```
   cd backend
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

### 2. Frontend Setup

1. **Install dependencies:**
   ```
   cd frontend
   npm install
   ```

3. **Install root-level dependencies (for concurrently):**
   ```
   # From project root directory
   npm install
   ```

## Running the Application

We've provided multiple ways to run the application:

### Option 1: Using concurrently (Recommended)

This method lets you run both the frontend and backend with a single command:

```
npm run both
```

This will start both servers simultaneously in the same terminal window, with clear output from each process.

### Option 2: Alternative concurrently syntax

If you prefer a different concurrently style:

```
npm run dev
```

### Option 3: Using the Provided Scripts

#### Windows PowerShell:
```
.\run-dev.ps1
```

#### Windows Command Prompt:
```
run-dev.bat
```

These scripts will:
- Ensure necessary directories exist
- Start the backend server (Python/FastAPI)
- Start the frontend development server (React/Vite)
- Open separate windows for each service

### Option 4: Running Manually

If you prefer to run the services separately:

#### Terminal 1 (Backend):
```
cd backend
python main.py
```

#### Terminal 2 (Frontend):
```
cd frontend
npm run dev
```

## Accessing the Application

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## Troubleshooting

### Concurrently Issues

If you encounter issues with the concurrently approach:
- Make sure you've installed the root-level dependencies: `npm install`
- Try running the services manually in separate terminals (Option 4)

### Tailwind CSS Issues

If you encounter Tailwind CSS errors, try reinstalling the dependencies:

```
cd frontend
npm uninstall tailwindcss postcss autoprefixer
npm install -D tailwindcss@3.3.3 postcss@8.4.29 autoprefixer@10.4.15
npx tailwindcss init -p
```

### Backend Dependencies

If you have issues with Python dependencies:

```
pip install --upgrade -r requirements.txt
```

### Port Conflicts

If either port 5173 or 8000 is already in use:

- For the backend, modify the port in `backend/main.py`
- For the frontend, you can specify a different port with:
  ```
  cd frontend
  npm run dev -- --port 3000
  ```

## Further Help

Refer to the README.md file for more information about the project structure and features. 