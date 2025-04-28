# Bitcoin Price Prediction Frontend

This is the frontend component of the Bitcoin Price Prediction project. It provides a user interface for viewing Bitcoin price predictions, comparing models, and backtesting trading strategies.

## Features

- Real-time Bitcoin price display
- Price prediction visualization
- Model comparison tools
- Backtesting capabilities
- Dark mode support
- Responsive design

## Project Structure

- `src/`: Source code
  - `components/`: Reusable UI components
  - `pages/`: Application pages
  - `context/`: React context providers
  - `services/`: API services

## Getting Started

### Prerequisites

- Node.js 14.0 or higher
- npm or yarn

### Installation

1. Clone the repository
2. Navigate to the frontend directory
3. Install dependencies:

```bash
npm install
# or
yarn
```

### Running the Development Server

```bash
npm run dev
# or
yarn dev
```

The application will be available at `http://localhost:5173`

### Building for Production

```bash
npm run build
# or
yarn build
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
VITE_API_URL=http://localhost:8000/api
```

## Available Scripts

- `npm run dev` - Start the development server
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run preview` - Preview the production build locally

## Technologies Used

- React
- React Router
- Chart.js
- Tailwind CSS
- Vite
