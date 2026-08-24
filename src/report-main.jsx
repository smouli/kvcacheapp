import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import Report from './Report.jsx'
import './Report.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Report />
  </StrictMode>,
)
