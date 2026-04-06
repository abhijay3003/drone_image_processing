/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, Type } from "@google/genai";
import { 
  Droplets, 
  Leaf, 
  AlertTriangle, 
  CheckCircle2, 
  Upload, 
  TrendingUp, 
  Map as MapIcon,
  Info,
  ChevronRight,
  Wind,
  Sun,
  CloudRain,
  Flame,
  Activity,
  Zap,
  Target,
  Layers,
  Thermometer,
  Database
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

// --- TYPES ---

interface AnalysisResult {
  healthy: number;
  stressed: number;
  critical: number;
  risks: Risk[];
  recommendations: string[];
  yield: YieldData;
  alerts: AlertArea[];
  soilMoisture: number;
  nitrogenLevel: number;
  pestRisk: number;
  criticalZones: CriticalZone[];
  disasterRisks: DisasterRisk[];
  detectedDisasterType?: string;
  metadata: {
    uvIndex: number;
    humidity: number;
    gsd: number;
    altitude: number;
    temp: number;
    timestamp: string;
    soilPh: number;
    lai: number; // Leaf Area Index
    chlorophyll: number; // mg/m2
  };
}

interface DisasterRisk {
  id: string;
  type: 'Flood' | 'Fire' | 'Storm' | 'Erosion' | 'Volcanic' | 'Earthquake';
  location: string;
  probability: number;
  impact: string;
  warning: string;
  severity: 'Low' | 'Medium' | 'High' | 'Critical';
}

interface CriticalZone {
  id: string;
  location: string;
  severity: 'Critical' | 'High';
  analysis: string;
  reasons: string[];
  action: string;
}

interface YieldData {
  expected: string;
  confidence: number;
  reasons: string[];
  details: string;
}

interface AlertArea {
  id: string;
  label: string;
  issue: string;
  reason: string;
  details: string;
  severity: 'High' | 'Medium' | 'Low' | 'Critical';
  pos: { x: number, y: number };
}

interface Risk {
  type: string;
  severity: 'High' | 'Medium' | 'Low' | 'Critical';
  reason: string;
}

// --- SAMPLE DATA ---

const SAMPLE_IMAGES = [
  {
    id: 'field-1',
    name: 'Healthy Wheat Field',
    url: 'https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&q=80&w=1000',
    description: 'Optimal growth conditions, high greenery.'
  },
  {
    id: 'field-2',
    name: 'Drought Stressed Corn',
    url: 'https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=1000',
    description: 'Visible yellowing and dry patches.'
  },
  {
    id: 'field-3',
    name: 'Water Logged Sector',
    url: 'https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&q=80&w=1000',
    description: 'Dark, saturated soil with sparse vegetation.'
  },
  {
    id: 'flood-1',
    name: 'Severe Flood Zone',
    url: 'https://images.unsplash.com/photo-1547683905-f686c993aae5?auto=format&fit=crop&q=80&w=1000',
    description: 'Widespread inundation in urban/rural interface.'
  },
  {
    id: 'lava-1',
    name: 'Active Lava Flow',
    url: 'https://images.unsplash.com/photo-1518457607834-6e8d80c183c5?auto=format&fit=crop&q=80&w=1000',
    description: 'Molten rock and volcanic activity.'
  },
  {
    id: 'quake-1',
    name: 'Seismic Debris',
    url: 'https://images.unsplash.com/photo-1516733725897-1aa73b87c8e8?auto=format&fit=crop&q=80&w=1000',
    description: 'Structural damage and debris field.'
  }
];

/**
 * Predicts if an image contains lava, flood, or earthquake hazards.
 * @param base64Image - The base64 encoded image string (without data:image/jpeg;base64, prefix)
 */
async function predictHazard(base64Image: string) {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  
  const response = await ai.models.generateContent({
    model: "gemini-3-flash-preview",
    contents: [
      {
        parts: [
          { text: "Analyze this image. Detect if it shows a 'flood', 'lava', or 'earthquake'. Return JSON." },
          { inlineData: { data: base64Image, mimeType: "image/jpeg" } }
        ]
      }
    ],
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          hazard: { 
            type: Type.STRING, 
            enum: ["flood", "lava", "earthquake", "none"],
            description: "The type of hazard detected"
          },
          confidence: { type: Type.NUMBER },
          details: { type: Type.STRING }
        },
        required: ["hazard", "confidence", "details"]
      }
    }
  });

  return JSON.parse(response.text);
}

export default function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState<'original' | 'health' | 'risk'>('original');
  const [activeMode, setActiveMode] = useState<'agriculture' | 'risk'>('agriculture');
  const [selectedSector, setSelectedSector] = useState<AlertArea | null>(null);
  const [showUploadPrompt, setShowUploadPrompt] = useState(false);
  const [showFullReport, setShowFullReport] = useState(false);
  const [hoverData, setHoverData] = useState<{ x: number, y: number, r: number, g: number, b: number, label: string } | null>(null);
  const [riskFilterType, setRiskFilterType] = useState<'All' | 'Flood' | 'Fire' | 'Storm' | 'Erosion' | 'Volcanic' | 'Earthquake'>('All');
  const [riskFilterSeverity, setRiskFilterSeverity] = useState<'All' | 'Low' | 'Medium' | 'High' | 'Critical'>('All');
  
  // New Innovative Features State
  const [irrigationBoost, setIrrigationBoost] = useState(20);
  const [fertilizerBoost, setFertilizerBoost] = useState(10);
  const [showPriorityMap, setShowPriorityMap] = useState(false);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setSelectedImage(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // --- ANALYSIS LOGIC (Simulated via Canvas) ---

  const calculateRiskSeverity = (probability: number, impact: string): 'Low' | 'Medium' | 'High' | 'Critical' => {
    let impactScore = 1;
    const lowerImpact = impact.toLowerCase();
    if (lowerImpact.includes('critical')) impactScore = 4;
    else if (lowerImpact.includes('major')) impactScore = 3;
    else if (lowerImpact.includes('medium')) impactScore = 2;

    const score = (probability / 100) * impactScore;
    
    if (score >= 2.5) return 'Critical';
    if (score >= 1.5) return 'High';
    if (score >= 0.6) return 'Medium';
    return 'Low';
  };

  const runAnalysis = async () => {
    if (!imageRef.current || !canvasRef.current) return;
    
    setIsAnalyzing(true);
    
    // Small delay to ensure UI updates
    await new Promise(resolve => setTimeout(resolve, 500));

    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    const img = imageRef.current!;
    
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    let healthyCount = 0;
    let stressedCount = 0;
    let criticalCount = 0;
    let waterCount = 0;
    let charredCount = 0;
    let lavaCount = 0;
    let debrisCount = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      // Refined color classification
      const isLava = r > 200 && g > 50 && g < 150 && b < 70; // High-intensity orange/red (Lava)
      
      // Stricter Water Detection
      const isBlueWater = b > r * 1.2 && b > g * 1.1 && b > 60; // Clear Blue Water
      const isMuddyWater = r > 60 && r < 140 && g > 55 && g < 130 && b > 45 && b < 110 && 
                          Math.abs(r - g) < 15 && g > b && (r - b) < 40; // Muddy Water (Neutral Brown)
      const isWater = isBlueWater || isMuddyWater;
      
      const isGreen = g > r * 1.1 && g > b * 1.1; // Healthy Vegetation
      const isDark = r < 35 && g < 35 && b < 35; // Charred/Deep Shadows
      const isYellow = r > 150 && g > 150 && b < 140 && r > b * 1.3; // Stressed
      
      // Debris is strictly neutral grey/concrete (Earthquake)
      const isDebris = !isWater && !isLava && Math.abs(r - g) < 8 && Math.abs(g - b) < 8 && r > 90 && r < 180;

      if (isLava) {
        lavaCount++;
      } else if (isWater) {
        waterCount++;
      } else if (isGreen) {
        healthyCount++;
      } else if (isDebris) {
        debrisCount++;
      } else if (isDark) {
        charredCount++;
      } else if (isYellow) {
        stressedCount++;
      } else {
        // Context-aware fallback
        if (activeMode === 'agriculture') {
          // If it's somewhat green, count as healthy but maybe slightly stressed
          if (g > r * 0.95 && g > b * 0.95) {
            healthyCount++;
          } else {
            criticalCount++;
          }
        } else {
          criticalCount++;
        }
      }
    }
    
    const total = healthyCount + stressedCount + criticalCount + waterCount + charredCount + lavaCount + debrisCount;
    const hPct = Math.round((healthyCount / total) * 100);
    const sPct = Math.round((stressedCount / total) * 100);
    const wPct = Math.round((waterCount / total) * 100);
    const chPct = Math.round((charredCount / total) * 100);
    const lPct = Math.round((lavaCount / total) * 100);
    const dPct = Math.round((debrisCount / total) * 100);
    const cPct = 100 - (hPct + sPct + wPct + chPct + lPct + dPct);
    
    let detectedDisasterType = 'None Detected';
    
    if (activeMode === 'risk') {
      try {
        const base64Image = canvas.toDataURL('image/jpeg').split(',')[1];
        const prediction = await predictHazard(base64Image);
        
        if (prediction.hazard === 'flood') detectedDisasterType = 'Flash Flood / Inundation';
        else if (prediction.hazard === 'lava') detectedDisasterType = 'Volcanic Eruption / Lava Flow';
        else if (prediction.hazard === 'earthquake') detectedDisasterType = 'Seismic Event / Earthquake';
        else if (chPct > 15) detectedDisasterType = 'Wildfire / Burn Zone';
        
        // Fallback to pixel analysis if Gemini says none but pixels say otherwise
        if (detectedDisasterType === 'None Detected') {
          const scores = [
            { type: 'Volcanic Eruption / Lava Flow', score: lPct * 2.5 },
            { type: 'Flash Flood / Inundation', score: wPct * 1.2 },
            { type: 'Seismic Event / Earthquake', score: dPct * 1.5 },
            { type: 'Wildfire / Burn Zone', score: chPct * 2.0 }
          ];
          const topDisaster = scores.sort((a, b) => b.score - a.score)[0];
          if (topDisaster.score > 12) {
            detectedDisasterType = topDisaster.type;
          }
        }
      } catch (error) {
        console.error("Gemini analysis failed:", error);
        // Fallback to pixel analysis
        const scores = [
          { type: 'Volcanic Eruption / Lava Flow', score: lPct * 2.5 },
          { type: 'Flash Flood / Inundation', score: wPct * 1.2 },
          { type: 'Seismic Event / Earthquake', score: dPct * 1.5 },
          { type: 'Wildfire / Burn Zone', score: chPct * 2.0 }
        ];
        const topDisaster = scores.sort((a, b) => b.score - a.score)[0];
        if (topDisaster.score > 12) {
          detectedDisasterType = topDisaster.type;
        }
      }
    }

    // --- TAILORED MATTER GENERATION ---
    let tailoredRisks: Risk[] = [];
    let tailoredRecs: string[] = [];
    let tailoredCriticalZones: CriticalZone[] = [];
    let tailoredAlerts: AlertArea[] = [];
    let tailoredYield: YieldData;

    const locations = ['North-East', 'Middle-West', 'South-West', 'Central', 'North-West', 'Eastern Boundary', 'Southern Slope', 'Perimeter A-4', 'Western Ridge', 'Gully Sector 7', 'Plateau Interface'];
    const loc1 = locations[hPct % locations.length];
    const loc2 = locations[(hPct + 5) % locations.length];

    if (activeMode === 'risk' && detectedDisasterType !== 'None Detected') {
      // Disaster-Specific Matter (STRICTLY NO COMBINATION)
      if (detectedDisasterType.includes('Flood')) {
        tailoredRisks = [{ type: 'Flash Flood Hazard', severity: 'Critical', reason: 'Surface water accumulation exceeding drainage capacity in low-lying catchment zones.' }];
        tailoredRecs = [
          'Clear all drainage channels and deploy temporary flood barriers.',
          'Monitor secondary channels for overflow and subsurface saturation levels.',
          'Activate emergency flood response protocols for the lower perimeter.'
        ];
        tailoredCriticalZones = [{
          id: 'CZ-01',
          location: 'Low-lying Catchment Basin',
          severity: 'Critical',
          analysis: 'Surface water accumulation exceeding drainage capacity in low-lying catchment zones.',
          reasons: ['Blocked culverts and drainage channels', 'Saturated subsoil (100% field capacity)', 'High precipitation accumulation (>50mm)'],
          action: 'Clear all drainage channels and deploy temporary flood barriers in the lower perimeter.'
        }];
        tailoredAlerts = [{
          id: 'A1',
          label: 'Flood Zone',
          issue: 'Hydraulic Pressure',
          reason: 'Subsurface saturation > 85%',
          details: 'High flood risk. Drainage capacity is reaching its limit. Monitor secondary channels for overflow.',
          severity: 'High',
          pos: { x: 30, y: 25 }
        }];
      } else if (detectedDisasterType.includes('Volcanic')) {
        tailoredRisks = [{ type: 'Volcanic Activity', severity: 'Critical', reason: 'Active lava flow or high-temperature magmatic signatures detected in the proximity.' }];
        tailoredRecs = [
          'Immediate evacuation of the exclusion zone.',
          'Monitor flow direction and velocity via thermal imaging.',
          'Coordinate with local emergency services for large-scale evacuation.'
        ];
        tailoredCriticalZones = [{
          id: 'CZ-01',
          location: 'Tectonic Interface Zone',
          severity: 'Critical',
          analysis: 'Active lava flow or high-temperature magmatic signatures detected in the proximity.',
          reasons: ['Extreme thermal IR signature (>800°C)', 'Molten rock flow patterns identified', 'Sulfur dioxide plume detection'],
          action: 'Immediate evacuation of the exclusion zone. Monitor flow direction and velocity.'
        }];
        tailoredAlerts = [
          {
            id: 'A1',
            label: 'Primary Vent',
            issue: 'Magmatic Activity',
            reason: 'Surface temperature > 800°C',
            details: 'Active lava flow detected. Immediate evacuation required.',
            severity: 'Critical',
            pos: { x: 45, y: 40 }
          },
          {
            id: 'A2',
            label: 'Flow Perimeter',
            issue: 'Thermal Propagation',
            reason: 'Anomalous heat signature detected',
            details: 'Lava flow is expanding towards the southern sector.',
            severity: 'High',
            pos: { x: 55, y: 55 }
          }
        ];
      } else if (detectedDisasterType.includes('Seismic')) {
        tailoredRisks = [{ type: 'Seismic Damage', severity: 'Critical', reason: 'Significant structural debris and ground fissures indicating recent seismic activity.' }];
        tailoredRecs = [
          'Activate search and rescue protocols immediately.',
          'Secure unstable structures and check utility lines for leaks.',
          'Establish safe zones for displaced individuals and monitor for aftershocks.'
        ];
        tailoredCriticalZones = [{
          id: 'CZ-01',
          location: 'Fault Line Perimeter',
          severity: 'Critical',
          analysis: 'Significant structural debris and ground fissures indicating recent seismic activity.',
          reasons: ['Widespread structural collapse detected', 'Large-scale ground surface displacement', 'Secondary debris flow in urban sectors'],
          action: 'Activate search and rescue protocols. Secure unstable structures and check utility lines.'
        }];
        tailoredAlerts = [
          {
            id: 'A1',
            label: 'Epicenter Zone',
            issue: 'Structural Collapse',
            reason: 'Widespread ground rupture detected',
            details: 'Significant structural damage observed. High risk of further collapse.',
            severity: 'Critical',
            pos: { x: 50, y: 50 }
          },
          {
            id: 'A2',
            label: 'Debris Field',
            issue: 'Blocked Access',
            reason: 'Secondary debris flow detected',
            details: 'Roads and access points are blocked by structural debris.',
            severity: 'High',
            pos: { x: 35, y: 65 }
          }
        ];
      } else if (detectedDisasterType.includes('Wildfire')) {
        tailoredRisks = [{ type: 'Wildfire Propagation', severity: 'Critical', reason: 'High fuel load density and extreme thermal signatures detected in dry biomass zones.' }];
        tailoredRecs = [
          'Create firebreaks and perform controlled removal of dead organic matter.',
          'Monitor boundary moisture gradients and wind speed/direction.',
          'Deploy automated suppression systems in high-risk corridors.'
        ];
        tailoredCriticalZones = [{
          id: 'CZ-01',
          location: 'Dry Perimeter Boundary',
          severity: 'Critical',
          analysis: 'High fuel load density and extreme thermal signatures detected in dry biomass zones.',
          reasons: ['Dry biomass accumulation (>5kg/m²)', 'Relative humidity below 20%', 'Surface temperature exceeds 50°C'],
          action: 'Create firebreaks and perform controlled removal of dead organic matter (fuel reduction).'
        }];
        tailoredAlerts = [{
          id: 'A1',
          label: 'Fire Zone',
          issue: 'Critical Fuel Load',
          reason: 'Biomass density exceeds 5kg/m²',
          details: 'Extreme wildfire risk. High accumulation of dry organic matter detected.',
          severity: 'High',
          pos: { x: 30, y: 25 }
        }];
      }
      
      tailoredYield = {
        expected: `${(100 - cPct)}% Structural Integrity`,
        confidence: Math.min(98, 80 + (100 - cPct) / 5),
        reasons: [
          "Disaster event detected in primary quadrants",
          "Infrastructure integrity compromised by environmental hazard",
          "Drainage network capacity under extreme stress",
          "Thermal signatures exceed safety thresholds"
        ],
        details: `The site assessment is dominated by the detected ${detectedDisasterType}. Structural integrity is currently estimated at ${100 - cPct}%. Primary focus is on life safety and asset protection in the affected zones.`
      };
    } else {
      // Agriculture Mode or No Disaster Detected in Risk Mode
      if (activeMode === 'agriculture') {
        if (sPct > 15) tailoredRisks.push({ type: 'Drought Stress', severity: 'High', reason: 'Widespread stomatal closure and foliage yellowing detected.' });
        if (wPct > 10) tailoredRisks.push({ type: 'Soil Anoxia', severity: 'Medium', reason: 'Prolonged saturation inhibiting root respiration.' });
        if (hPct < 60) tailoredRisks.push({ type: 'Yield Deficit', severity: 'Critical', reason: 'Insufficient healthy biomass to meet seasonal targets.' });
        
        tailoredRecs = [
          sPct > 10 ? 'Initiate precision irrigation in the identified stress zones.' : 'Maintain current hydration schedule.',
          cPct > 15 ? 'Apply organic soil conditioners to improve drainage and aeration.' : 'Continue soil health monitoring.',
          'Schedule a high-resolution multispectral scan for next week.'
        ];

        tailoredAlerts = [
          {
            id: 'A1',
            label: `${loc1} Sector`,
            issue: sPct > 15 ? 'Hydration Deficit' : 'Nitrogen Imbalance',
            reason: sPct > 15 ? 'Thermal IR signature indicates stomatal closure' : 'Spectral reflectance drop in 700nm range',
            details: sPct > 15 
              ? 'Critical water stress detected. Immediate irrigation override required for this sector.'
              : 'Early signs of chlorosis detected. Recommended application of liquid nitrogen supplement.',
            severity: sPct > 25 ? 'High' : 'Medium',
            pos: { x: 25, y: 30 }
          },
          {
            id: 'A2',
            label: `${loc2} Sector`,
            issue: cPct > 10 ? 'Pathogen Proliferation' : 'Thermal Overload',
            reason: cPct > 10 ? 'Localized NDVI drop and necrotic spotting' : 'Surface temperature 5°C above safety threshold',
            details: cPct > 10
              ? 'Biological threat detected. Immediate isolation and targeted application of organic fungicide is required.'
              : 'Extreme heat stress detected. Recommend activating automated shading systems.',
            severity: cPct > 15 ? 'High' : 'Medium',
            pos: { x: 65, y: 60 }
          }
        ];

        const agriIssues = [
          {
            id: 'nutrient',
            name: 'Nutrient Deficiency',
            analysis: 'Spectral shift indicates significant nitrogen and phosphorus imbalance in the upper canopy.',
            reasons: ['Low chlorophyll reflectance (700-750nm)', 'Stunted growth patterns in central rows', 'Interveinal chlorosis detected'],
            action: 'Apply targeted NPK fertilization (20-10-10) and perform soil pH adjustment to 6.5.',
            weight: cPct + sPct * 0.5
          },
          {
            id: 'disease',
            name: 'Disease Outbreak',
            analysis: 'Necrotic spotting and rapid foliage degradation detected, suggesting a fungal pathogen spread.',
            reasons: ['Fungal mycelium signatures in thermal IR', 'High humidity levels in the lower canopy', 'Rapid NDVI drop over 48 hours'],
            action: 'Apply broad-spectrum organic fungicide and increase row spacing to improve airflow.',
            weight: cPct * 1.2
          }
        ];
        const selectedAgriIssues = [...agriIssues].sort((a, b) => b.weight - a.weight).slice(0, 2);
        tailoredCriticalZones = selectedAgriIssues.map((issue, idx) => ({
          id: `CZ-0${idx + 1}`,
          location: idx === 0 ? `${loc1} Quadrant` : `${loc2} Slope Interface`,
          severity: (idx === 0 && issue.weight > 20) ? 'Critical' : 'High',
          analysis: issue.analysis,
          reasons: issue.reasons,
          action: issue.action
        }));

        const yieldValue = (hPct * 0.85).toFixed(1);
        tailoredYield = {
          expected: `${yieldValue} Tons per Acre`,
          confidence: Math.min(98, 85 + (hPct / 10)),
          reasons: [
            hPct > 70 ? "Optimal chlorophyll absorption in the upper canopy" : "Patchy vegetation density detected",
            "Uniform plant height across primary growth sectors",
            "Soil moisture retention within target parameters",
            "Low thermal stress signature in central quadrants"
          ],
          details: `The field is currently exhibiting ${hPct > 75 ? 'exceptional' : 'moderate'} biological vigor, reaching ${hPct}% of its theoretical maximum health.`
        };
      } else {
        // Risk Mode - No Disaster
        if (wPct > 15) tailoredRisks.push({ type: 'Flash Flood Hazard', severity: 'High', reason: 'Surface runoff capacity exceeded in secondary channels.' });
        if (chPct > 10) tailoredRisks.push({ type: 'Wildfire Propagation', severity: 'Critical', reason: 'High-density dry fuel load in high-wind corridors.' });
        if (cPct > 8) tailoredRisks.push({ type: 'Foundation Erosion', severity: 'Medium', reason: 'Soil displacement detected near load-bearing structures.' });

        tailoredRecs = [
          cPct > 8 ? 'Deploy geotechnical stabilization measures on the western slope.' : 'Maintain existing drainage clearways.',
          sPct > 12 ? 'Perform controlled clearing of dry biomass in the perimeter.' : 'Monitor boundary moisture gradients.',
          'Conduct a structural integrity audit of all primary foundations.'
        ];

        tailoredAlerts = [
          {
            id: 'A1',
            label: `${loc1} Boundary`,
            issue: wPct > 10 ? 'Hydraulic Pressure' : (chPct > 8 ? 'Critical Fuel Load' : 'Surface Erosion'),
            reason: wPct > 10 ? 'Subsurface saturation > 85%' : (chPct > 8 ? 'Biomass density exceeds 5kg/m²' : 'Lack of stabilizing ground cover'),
            details: wPct > 10
              ? 'High flood risk. Drainage capacity is reaching its limit. Monitor secondary channels for overflow.'
              : (chPct > 8 ? 'Extreme wildfire risk. High accumulation of dry organic matter detected.' : 'Active topsoil loss detected due to lack of root stabilization.'),
            severity: (wPct > 20 || chPct > 15) ? 'High' : 'Medium',
            pos: { x: 30, y: 25 }
          },
          {
            id: 'A2',
            label: `${loc2} Slope`,
            issue: cPct > 12 ? 'Geotechnical Instability' : (wPct > 15 ? 'Erosion Corridor' : 'Thermal Anomaly'),
            reason: cPct > 12 ? 'Soil shear strength failure detected' : (wPct > 15 ? 'Active sediment transport detected' : 'Anomalous heat signature in boundary zone'),
            details: cPct > 12
              ? 'High landslide risk. Slope saturation levels have reached critical shear threshold.'
              : (wPct > 15 ? 'Active topsoil loss detected. Surface runoff is carving deep channels.' : 'Unusual heat signature detected. Potential ignition source or localized heating.'),
            severity: (cPct > 20 || wPct > 25) ? 'High' : 'Medium',
            pos: { x: 70, y: 65 }
          }
        ];

        const riskIssues = [
          {
            id: 'structural',
            name: 'Structural Instability',
            analysis: 'Foundation displacement and micro-fissures in load-bearing soil detected near primary structures.',
            reasons: ['Soil shear strength failure detected', 'Micro-seismic vibrations in foundation', 'Visible cracks in structural subsoil'],
            action: 'Evacuate immediate vicinity and install structural shoring to prevent collapse.',
            weight: cPct * 1.5
          },
          {
            id: 'erosion',
            name: 'Erosion Hazard',
            analysis: 'Soil shear strength failure on steep gradient due to lack of stabilizing vegetation.',
            reasons: ['Lack of root stabilization in topsoil', 'High surface runoff velocity (>2m/s)', 'Active sediment transport in gully'],
            action: 'Install silt fences, hydroseed for rapid cover, and divert surface runoff away from slope.',
            weight: (wPct * 0.5) + (cPct * 0.5)
          }
        ];
        const selectedRiskIssues = [...riskIssues].sort((a, b) => b.weight - a.weight).slice(0, 2);
        tailoredCriticalZones = selectedRiskIssues.map((issue, idx) => ({
          id: `CZ-0${idx + 1}`,
          location: idx === 0 ? `${loc1} Quadrant` : `${loc2} Slope Interface`,
          severity: (idx === 0 && issue.weight > 20) ? 'Critical' : 'High',
          analysis: issue.analysis,
          reasons: issue.reasons,
          action: issue.action
        }));

        tailoredYield = {
          expected: `${(100 - cPct)}% Structural Integrity`,
          confidence: Math.min(98, 80 + (100 - cPct) / 5),
          reasons: [
            cPct < 10 ? "Ground surface stability index is optimal" : "Localized soil displacement detected",
            "Infrastructure foundations show no thermal anomalies",
            "Drainage network capacity exceeds current saturation",
            "Vegetation moisture levels above critical ignition threshold"
          ],
          details: `The comprehensive site safety assessment indicates a ${cPct < 15 ? 'stable' : 'monitored'} status, with a ${100 - cPct}% integrity rating.`
        };
      }
    }

    const disasterRisks: DisasterRisk[] = [
      {
        id: 'DR-01',
        type: 'Flood',
        location: 'Low-lying Catchment Basin',
        probability: Math.min(95, wPct * 3 + 5),
        impact: activeMode === 'agriculture' ? 'Critical - Potential total crop loss and soil nutrient washout' : 'Critical - Potential structural damage to primary infrastructure',
        warning: 'Anomalous water accumulation detected in deep-soil saturation sensors.',
        severity: 'Low'
      },
      {
        id: 'DR-02',
        type: 'Fire',
        location: 'Dry Perimeter Boundary',
        probability: Math.min(95, chPct * 4 + sPct),
        impact: activeMode === 'agriculture' ? 'Major - Risk to storage facilities and equipment' : 'Major - Risk to boundary security and utility lines',
        warning: 'Thermal anomalies and charred biomass detected in boundary vegetation.',
        severity: 'Low'
      },
      {
        id: 'DR-05',
        type: 'Volcanic',
        location: 'Tectonic Interface Zone',
        probability: Math.min(95, lPct * 5),
        impact: 'Critical - Total destruction of assets and life threat',
        warning: 'Magmatic thermal signatures detected in surface scans.',
        severity: 'Low'
      },
      {
        id: 'DR-06',
        type: 'Earthquake',
        location: 'Fault Line Perimeter',
        probability: Math.min(95, dPct * 2 + cPct),
        impact: 'Critical - Widespread structural failure and ground rupture',
        warning: 'Significant debris accumulation and ground fissures detected.',
        severity: 'Low'
      }
    ].filter(risk => {
      if (detectedDisasterType !== 'None Detected') {
        return detectedDisasterType.includes(risk.type);
      }
      if (risk.type === 'Flood') return wPct > 5;
      if (risk.type === 'Fire') return chPct > 5 || (sPct > 20 && activeMode === 'agriculture');
      if (risk.type === 'Volcanic') return lPct > 2;
      if (risk.type === 'Earthquake') return dPct > 10 || cPct > 15;
      return true;
    }).map(risk => ({
      ...risk,
      severity: calculateRiskSeverity(risk.probability, risk.impact)
    })) as DisasterRisk[];

    const humidity = Math.min(100, 30 + wPct * 2 + (hPct / 4));
    
    setResult({
      healthy: hPct,
      stressed: sPct,
      critical: cPct,
      risks: tailoredRisks,
      recommendations: tailoredRecs,
      yield: tailoredYield!,
      alerts: tailoredAlerts,
      soilMoisture: Math.min(100, 20 + wPct * 2 + (hPct / 5)),
      nitrogenLevel: Math.min(100, 40 + (hPct / 2) - (sPct / 2)),
      pestRisk: Math.min(100, 5 + cPct + (humidity / 10)),
      criticalZones: tailoredCriticalZones,
      disasterRisks,
      detectedDisasterType,
      metadata: {
        uvIndex: Math.min(11, 3 + (sPct / 10)),
        humidity: humidity,
        gsd: 1.2,
        altitude: 120,
        temp: 22 + (sPct / 5) + (chPct / 2),
        timestamp: new Date().toISOString(),
        soilPh: 6.2 + (cPct / 100),
        lai: 0.5 + (hPct / 25),
        chlorophyll: 150 + (hPct * 3)
      }
    });
    setIsAnalyzing(false);
  };

  useEffect(() => {
    if (selectedImage) {
      setResult(null);
      setActiveTab('original');
    }
  }, [selectedImage]);

  // --- RENDER HEALTH MAP ---
  useEffect(() => {
    if (result && activeTab !== 'original' && canvasRef.current && imageRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d')!;
      const img = imageRef.current;
      
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        
        const isGreen = g > r * 1.02 && g > b * 1.02;
        const isBlue = b > r * 1.1 && b > g * 1.1;
        const isDark = r < 40 && g < 40 && b < 40;
        const isYellow = r > 130 && g > 130 && b < 150;

        if (activeTab === 'health') {
          if (showPriorityMap) {
            // Priority Map Logic: Highlight critical/stressed zones in high-risk areas
            const isHighRisk = (result.critical > 20 || result.stressed > 30);
            if (isHighRisk && (isYellow || (!isGreen && !isBlue))) {
              data[i] = 255; data[i+1] = 0; data[i+2] = 0; // Bright Red for Priority
            } else if (isGreen) {
              data[i] = 34; data[i+1] = 197; data[i+2] = 94; // Green
            } else {
              data[i] = 50; data[i+1] = 50; data[i+2] = 50; // Dim others
            }
          } else {
            if (isGreen) {
              data[i] = 34; data[i+1] = 197; data[i+2] = 94; // Green
            } else if (isYellow) {
              data[i] = 234; data[i+1] = 179; data[i+2] = 8; // Yellow
            } else {
              data[i] = 239; data[i+1] = 68; data[i+2] = 68; // Red
            }
          }
        } else if (activeTab === 'risk') {
          // Enhanced Risk Coloring: Highlight hazards with high-contrast neon colors
          // ONLY apply coloring for Flood as requested. Lava and Earthquake are detected but not colored separately.
          if (result.detectedDisasterType?.includes('Flood')) {
            if (isBlue) {
              // Flood Hazard - Neon Cyan/Blue
              data[i] = 34; data[i+1] = 211; data[i+2] = 238; 
            } else if (isGreen) {
              // Healthy areas - Desaturated to emphasize risks
              const avg = (r + g + b) / 3;
              data[i] = avg * 0.5; data[i+1] = avg * 0.5; data[i+2] = avg * 0.5; 
            } else {
              // Other critical areas - Dark Red
              data[i] = 153; data[i+1] = 27; data[i+2] = 27; 
            }
          } else {
            // No special coloring for Lava/Earthquake/Other in Risk mode
            // Just show original image pixels (already drawn by ctx.drawImage)
            // We don't need to do anything here as ctx.drawImage(img, 0, 0) was called at the start
          }
        }
      }
      ctx.putImageData(imageData, 0, 0);
    }
  }, [activeTab, result]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!canvasRef.current || activeTab === 'original' || !result) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    
    // Calculate canvas coordinates
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const canvasX = x * scaleX;
    const canvasY = y * scaleY;
    
    try {
      const pixel = ctx.getImageData(canvasX, canvasY, 1, 1).data;
      const [r, g, b] = pixel;
      
      let label = 'Unknown';
      if (activeTab === 'health') {
        if (r === 34 && g === 197 && b === 94) label = 'Healthy Vegetation';
        else if (r === 234 && g === 179 && b === 8) label = 'Stressed / Dry';
        else if (r === 239 && g === 68 && b === 68) label = 'Critical / Bare Ground';
        else label = 'Original Terrain';
      } else if (activeTab === 'risk') {
        if (result.detectedDisasterType?.includes('Flood')) {
          if (r === 34 && g === 211 && b === 238) label = 'Flash Flood / Inundation';
          else if (r === 153 && g === 27 && b === 27) label = 'Critical Hazard Zone';
          else label = 'Analyzing Flood Impact...';
        } else if (result.detectedDisasterType?.includes('Volcanic')) {
          label = 'Analyzing Volcanic Activity...';
        } else if (result.detectedDisasterType?.includes('Seismic')) {
          label = 'Analyzing Seismic Damage...';
        } else if (result.detectedDisasterType?.includes('Wildfire')) {
          label = 'Analyzing Fire Propagation...';
        } else {
          label = 'Scanning for Hazards...';
        }
      }
      
      setHoverData({ x, y, r, g, b, label });
    } catch (err) {
      // Ignore out of bounds
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0c10] text-gray-100 font-sans selection:bg-emerald-500/30">
      {/* Header */}
      <header className="border-b border-white/5 bg-black/40 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-emerald-500 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Wind className="text-white w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">AgroDrone AI</h1>
              <p className="text-[10px] text-emerald-500 font-mono uppercase tracking-widest">Precision Agriculture System</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-gray-400">
            <a href="#" className="hover:text-white transition-colors">Dashboard</a>
            <a href="#" className="hover:text-white transition-colors">Fleet Management</a>
            <a href="#" className="hover:text-white transition-colors">Historical Data</a>
            <button className="px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg border border-white/10 transition-all">
              Settings
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-10">
        {/* Main Mode Selector */}
        <div className="flex items-center gap-4 mb-8 bg-white/5 p-1.5 rounded-2xl border border-white/10 w-fit">
          <button 
            onClick={() => {
              setActiveMode('agriculture');
              setSelectedImage(null);
              setResult(null);
              setActiveTab('original');
            }}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-bold transition-all ${
              activeMode === 'agriculture' 
              ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/20' 
              : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <Leaf className="w-4 h-4" />
            Agriculture Analysis
          </button>
          <button 
            onClick={() => {
              setActiveMode('risk');
              setSelectedImage(null);
              setResult(null);
              setActiveTab('original');
            }}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-bold transition-all ${
              activeMode === 'risk' 
              ? 'bg-red-500 text-white shadow-lg shadow-red-500/20' 
              : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <AlertTriangle className="w-4 h-4" />
            Risk Area Prediction
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Image Selection & Preview */}
          <div className="lg:col-span-8 space-y-6">
            
            {/* Image Selector & Upload */}
            <div className="bg-white/5 border border-white/10 rounded-2xl p-6 overflow-hidden relative group">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-500 to-blue-500 opacity-50" />
              
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
                <div>
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Database className="w-5 h-5 text-emerald-500" />
                    Drone Data Source
                  </h2>
                  <p className="text-xs text-gray-500 mt-1">Select from fleet storage or upload external imagery</p>
                </div>
                
                <div className="flex items-center gap-2">
                  <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="flex-1 md:flex-none px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-white rounded-xl text-xs font-bold flex items-center justify-center gap-2 transition-all shadow-lg shadow-emerald-500/20"
                  >
                    <Upload className="w-4 h-4" />
                    Upload Custom Image
                  </button>
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    className="hidden" 
                    accept="image/*"
                    onChange={handleFileUpload}
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                {SAMPLE_IMAGES.map((img) => (
                  <button
                    key={img.id}
                    onClick={() => setSelectedImage(img.url)}
                    className={`relative rounded-xl overflow-hidden aspect-video border-2 transition-all group ${
                      selectedImage === img.url ? 'border-emerald-500 scale-[1.02]' : 'border-transparent opacity-60 hover:opacity-100'
                    }`}
                  >
                    <img src={img.url} alt={img.name} className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                    <div className="absolute inset-0 bg-black/40 flex items-end p-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <span className="text-[10px] font-medium">{img.name}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Main Viewport */}
            <div 
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHoverData(null)}
              className="bg-black border border-white/10 rounded-2xl overflow-hidden aspect-video relative flex items-center justify-center group shadow-2xl cursor-crosshair"
            >
              {selectedImage ? (
                <>
                  <img 
                    ref={imageRef}
                    src={selectedImage} 
                    alt="Drone View" 
                    className={`w-full h-full object-cover transition-opacity duration-500 ${activeTab === 'original' ? 'opacity-100' : 'opacity-0'}`}
                    referrerPolicy="no-referrer"
                    crossOrigin="anonymous"
                    onLoad={() => {}}
                  />
                  <canvas 
                    ref={canvasRef}
                    className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-500 ${activeTab !== 'original' ? 'opacity-100' : 'opacity-0'}`}
                  />

                  {/* Interactive Tooltip */}
                  {hoverData && (
                    <div 
                      className="absolute pointer-events-none z-50 bg-black/80 backdrop-blur-md border border-white/20 p-2 rounded-lg shadow-xl flex flex-col gap-1 min-w-[120px]"
                      style={{ 
                        left: hoverData.x + 15, 
                        top: hoverData.y + 15,
                        transform: (hoverData.x > 400) ? 'translateX(-100%)' : ''
                      }}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[10px] font-bold text-white uppercase">{hoverData.label}</span>
                        <div 
                          className="w-2 h-2 rounded-full border border-white/20" 
                          style={{ backgroundColor: `rgb(${hoverData.r}, ${hoverData.g}, ${hoverData.b})` }}
                        />
                      </div>
                      <div className="flex flex-col gap-0.5">
                        <div className="flex justify-between text-[8px] font-mono text-gray-400">
                          <span>RGB:</span>
                          <span>{hoverData.r}, {hoverData.g}, {hoverData.b}</span>
                        </div>
                        <div className="flex justify-between text-[8px] font-mono text-gray-400">
                          <span>COORD:</span>
                          <span>{Math.round(hoverData.x)}, {Math.round(hoverData.y)}</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Telemetry Overlay */}
                  {result && (
                    <div className="absolute top-4 left-4 right-4 flex justify-between pointer-events-none z-30">
                      <div className="bg-black/40 backdrop-blur-sm border border-white/10 p-2 rounded-lg flex flex-col gap-1">
                        <div className="flex items-center gap-2">
                          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                          <span className="text-[9px] font-mono text-emerald-500 uppercase tracking-tighter">Live Telemetry</span>
                        </div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
                          <div className="flex items-center gap-1.5">
                            <Thermometer className="w-2.5 h-2.5 text-gray-500" />
                            <span className="text-[10px] font-mono text-white">{result.metadata.temp}°C</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <Wind className="w-2.5 h-2.5 text-gray-500" />
                            <span className="text-[10px] font-mono text-white">{result.metadata.altitude}m</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <Layers className="w-2.5 h-2.5 text-gray-500" />
                            <span className="text-[10px] font-mono text-white">{result.metadata.gsd.toFixed(2)}cm</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <Info className="w-2.5 h-2.5 text-gray-500" />
                            <span className="text-[10px] font-mono text-white">{result.metadata.timestamp}</span>
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-col items-end gap-2">
                        <div className="bg-black/40 backdrop-blur-sm border border-white/10 p-2 rounded-lg flex flex-col items-end gap-1">
                          <span className="text-[9px] font-mono text-gray-500 uppercase">Signal Strength</span>
                          <div className="flex gap-0.5 items-end h-2">
                            <div className="w-1 h-1 bg-emerald-500" />
                            <div className="w-1 h-1.5 bg-emerald-500" />
                            <div className="w-1 h-2 bg-emerald-500" />
                            <div className="w-1 h-2.5 bg-emerald-500/30" />
                          </div>
                        </div>
                        
                        <div className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest border ${
                          result.critical > 20 
                            ? 'bg-red-500/20 border-red-500/50 text-red-500' 
                            : 'bg-orange-500/20 border-orange-500/50 text-orange-500'
                        }`}>
                          {activeMode === 'agriculture' ? 'Crop Risk: ' : 'Hazard Level: '}
                          {result.critical > 20 ? 'Critical' : 'Moderate'}
                        </div>

                        {activeMode === 'risk' && result.detectedDisasterType && (
                          <div className="bg-blue-500/20 border border-blue-500/50 px-3 py-1 rounded-full text-[10px] font-bold text-blue-400 uppercase tracking-widest">
                            Detected: {result.detectedDisasterType}
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Alert Area Markers */}
                  {result && (
                    (activeMode === 'agriculture' && activeTab === 'health') || 
                    (activeMode === 'risk' && activeTab === 'risk')
                  ) && result.alerts.map((alert) => (
                    <motion.button
                      key={alert.id}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      whileHover={{ scale: 1.2 }}
                      onClick={() => {
                        setSelectedSector(alert);
                        setShowUploadPrompt(true);
                      }}
                      style={{ left: `${alert.pos.x}%`, top: `${alert.pos.y}%` }}
                      className={`absolute -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full border-2 border-white shadow-2xl flex items-center justify-center z-20 ${
                        alert.severity === 'Critical' ? 'bg-red-600 animate-pulse shadow-[0_0_20px_rgba(220,38,38,0.5)]' : alert.severity === 'High' ? 'bg-red-500' : 'bg-orange-500'
                      }`}
                    >
                      <AlertTriangle className="w-4 h-4 text-white" />
                      <div className="absolute top-full mt-2 bg-black/80 backdrop-blur-md px-2 py-1 rounded text-[10px] whitespace-nowrap pointer-events-none border border-white/10">
                        {alert.issue}
                      </div>
                    </motion.button>
                  ))}

                  {/* Drone HUD Overlay (Risk Mode Only) */}
                  {activeTab === 'risk' && result && (
                    <div className="absolute inset-0 pointer-events-none z-10 overflow-hidden">
                      {/* Corner Brackets */}
                      <div className="absolute top-6 left-6 w-12 h-12 border-t-2 border-l-2 border-white/20" />
                      <div className="absolute top-6 right-6 w-12 h-12 border-t-2 border-r-2 border-white/20" />
                      <div className="absolute bottom-6 left-6 w-12 h-12 border-b-2 border-l-2 border-white/20" />
                      <div className="absolute bottom-6 right-6 w-12 h-12 border-b-2 border-r-2 border-white/20" />
                      
                      {/* Scanning Line */}
                      <motion.div 
                        animate={{ top: ['0%', '100%', '0%'] }}
                        transition={{ duration: 6, repeat: Infinity, ease: "linear" }}
                        className="absolute left-0 right-0 h-[1px] bg-white/10 shadow-[0_0_15px_rgba(255,255,255,0.3)]"
                      />
                      
                      {/* Crosshair */}
                      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 border border-white/10 rounded-full flex items-center justify-center">
                        <div className="w-1 h-1 bg-white/30 rounded-full" />
                        <div className="absolute w-6 h-[1px] bg-white/10 -left-8" />
                        <div className="absolute w-6 h-[1px] bg-white/10 -right-8" />
                        <div className="absolute h-6 w-[1px] bg-white/10 -top-8" />
                        <div className="absolute h-6 w-[1px] bg-white/10 -bottom-8" />
                      </div>

                      {/* HUD Data Accents */}
                      <div className="absolute bottom-20 left-8 flex flex-col gap-1">
                        <div className="h-0.5 w-12 bg-white/20" />
                        <div className="h-0.5 w-8 bg-white/10" />
                        <div className="h-0.5 w-16 bg-white/5" />
                      </div>
                    </div>
                  )}
                  
                  {/* Overlay Controls */}
                  <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-1 bg-black/60 backdrop-blur-md border border-white/10 p-1 rounded-full shadow-2xl">
                    <button 
                      onClick={() => setActiveTab('original')}
                      className={`px-4 py-2 rounded-full text-xs font-medium transition-all ${activeTab === 'original' ? 'bg-white text-black' : 'hover:bg-white/10'}`}
                    >
                      Original
                    </button>
                    <button 
                      onClick={() => setActiveTab('health')}
                      disabled={!result}
                      className={`px-4 py-2 rounded-full text-xs font-medium transition-all ${!result ? 'opacity-50 cursor-not-allowed' : activeTab === 'health' ? 'bg-emerald-500 text-white' : 'hover:bg-white/10'}`}
                    >
                      Health Map
                    </button>
                    {activeTab === 'health' && result && (
                      <button 
                        onClick={() => setShowPriorityMap(!showPriorityMap)}
                        className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-2 ${showPriorityMap ? 'bg-red-500 text-white' : 'hover:bg-white/10 text-gray-400'}`}
                      >
                        <Target className="w-3 h-3" />
                        Priority
                      </button>
                    )}
                    {activeMode === 'risk' && (
                      <button 
                        onClick={() => setActiveTab('risk')}
                        disabled={!result}
                        className={`px-4 py-2 rounded-full text-xs font-medium transition-all ${!result ? 'opacity-50 cursor-not-allowed' : activeTab === 'risk' ? 'bg-orange-500 text-white' : 'hover:bg-white/10'}`}
                      >
                        Risk Analysis
                      </button>
                    )}
                  </div>

                  {/* Analysis Trigger */}
                  {!result && !isAnalyzing && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-[2px]">
                      <button 
                        onClick={runAnalysis}
                        className="px-8 py-4 bg-emerald-500 hover:bg-emerald-400 text-white rounded-2xl font-bold shadow-xl shadow-emerald-500/20 flex items-center gap-3 transition-all active:scale-95"
                      >
                        <TrendingUp className="w-5 h-5" />
                        Run AI Diagnostics
                      </button>
                    </div>
                  )}

                  {/* Loading State */}
                  {isAnalyzing && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-md">
                      <div className="w-16 h-16 border-4 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin mb-4" />
                      <p className="text-emerald-500 font-mono text-sm animate-pulse">SEGMENTING SPECTRAL DATA...</p>
                    </div>
                  )}

                  {/* Legend */}
                  {result && activeTab === 'health' && (
                    <div className="absolute top-6 right-6 bg-black/60 backdrop-blur-md border border-white/10 p-4 rounded-xl space-y-2">
                      <div className="flex items-center gap-2 text-xs">
                        <div className="w-3 h-3 bg-emerald-500 rounded-full" />
                        <span>Healthy Vegetation</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs">
                        <div className="w-3 h-3 bg-yellow-500 rounded-full" />
                        <span>Stressed / Dry</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs">
                        <div className="w-3 h-3 bg-red-500 rounded-full" />
                        <span>Critical / Soil</span>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center space-y-4">
                  <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center mx-auto border border-white/10">
                    <MapIcon className="w-10 h-10 text-gray-600" />
                  </div>
                  <p className="text-gray-500 text-sm">Select a field sector to begin analysis</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Metrics & Insights */}
          <div className="lg:col-span-4 space-y-6">
            
            {/* Agriculture Mode Content */}
            {activeMode === 'agriculture' && (
              <>
                {/* Crop Yield Prediction Section */}
                <AnimatePresence mode="wait">
                  {result && (
                    <motion.div 
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="bg-white/5 border border-white/10 rounded-2xl p-6"
                    >
                      <div className="flex items-center justify-between mb-6">
                        <h3 className="text-sm font-mono text-emerald-500 uppercase tracking-widest">Crop Yield Forecast</h3>
                        <div className="flex items-center gap-2 bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20">
                          <TrendingUp className="w-4 h-4 text-emerald-500" />
                          <span className="text-xs font-bold text-emerald-500">{result.yield.confidence}% Confidence</span>
                        </div>
                      </div>
                      
                      <div className="space-y-6">
                        <div className="p-4 bg-white/5 rounded-xl border border-white/10 flex items-center justify-between">
                          <div>
                            <p className="text-gray-500 text-xs uppercase font-bold mb-1">Expected Output</p>
                            <p className="text-3xl font-bold text-white">{result.yield.expected}</p>
                          </div>
                          <div className="w-24 h-12">
                            <ResponsiveContainer width="100%" height="100%">
                              <AreaChart data={[
                                { v: 65 }, { v: 72 }, { v: 68 }, { v: 85 }, { v: 92 }
                              ]}>
                                <Area type="monotone" dataKey="v" stroke="#10b981" fill="#10b98133" />
                              </AreaChart>
                            </ResponsiveContainer>
                          </div>
                        </div>

                        <div className="grid grid-cols-3 gap-3">
                          <div className="p-3 bg-white/5 rounded-xl border border-white/10 text-center">
                            <Droplets className="w-4 h-4 text-blue-400 mx-auto mb-1" />
                            <p className="text-[10px] text-gray-500 uppercase">Moisture</p>
                            <p className="text-sm font-bold">{result.soilMoisture}%</p>
                          </div>
                          <div className="p-3 bg-white/5 rounded-xl border border-white/10 text-center">
                            <Zap className="w-4 h-4 text-yellow-400 mx-auto mb-1" />
                            <p className="text-[10px] text-gray-500 uppercase">Nitrogen</p>
                            <p className="text-sm font-bold">{result.nitrogenLevel}%</p>
                          </div>
                          <div className="p-3 bg-white/5 rounded-xl border border-white/10 text-center">
                            <Activity className="w-4 h-4 text-red-400 mx-auto mb-1" />
                            <p className="text-[10px] text-gray-500 uppercase">Pest Risk</p>
                            <p className="text-sm font-bold">{result.pestRisk}%</p>
                          </div>
                        </div>

                        <div className="p-4 bg-black/20 rounded-xl border border-white/5">
                          <p className="text-gray-500 text-xs uppercase font-bold mb-2">Minute Details</p>
                          <p className="text-sm text-gray-400 leading-relaxed italic">
                            "{result.yield.details}"
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Critical Growth Zone Analysis (Agriculture Mode) */}
                <AnimatePresence mode="wait">
                  {result && result.critical > 5 && (
                    <motion.div 
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="bg-white/5 border border-white/10 rounded-2xl p-6"
                    >
                      <div className="flex items-center justify-between mb-6">
                        <h3 className="text-sm font-mono text-red-500 uppercase tracking-widest">Critical Growth Zones</h3>
                        <div className="bg-red-500/10 px-2 py-1 rounded border border-red-500/20">
                          <span className="text-[10px] font-bold text-red-500 uppercase">Yield Loss Risk</span>
                        </div>
                      </div>
                      
                      <div className="space-y-4">
                        {result.criticalZones.map((zone) => (
                          <div key={zone.id} className="p-4 bg-red-500/5 border border-red-500/20 rounded-xl space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-bold text-red-500">{zone.location}</span>
                              <span className="text-[10px] text-red-400 font-bold uppercase">Critical</span>
                            </div>
                            <p className="text-xs text-gray-300 leading-tight">{zone.analysis}</p>
                            <div className="pt-2 flex items-center gap-2">
                              <Target className="w-3 h-3 text-red-500" />
                              <p className="text-[10px] text-red-400 font-bold uppercase tracking-wider">Action: {zone.action}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>


                {/* What-If Simulator (Agriculture Mode Only) */}
                <AnimatePresence mode="wait">
                  {result && activeMode === 'agriculture' && (
                    <motion.div 
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="bg-white/5 border border-white/10 rounded-2xl p-6"
                    >
                      <div className="flex items-center justify-between mb-6">
                        <h3 className="text-sm font-mono text-blue-500 uppercase tracking-widest">What-If Simulator</h3>
                        <div className="bg-blue-500/10 px-2 py-1 rounded border border-blue-500/20">
                          <span className="text-[10px] font-bold text-blue-500 uppercase">Resource Simulation</span>
                        </div>
                      </div>
                      
                      <div className="space-y-6">
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <div className="flex justify-between text-[10px] font-bold text-gray-500 uppercase">
                              <span>Additional Irrigation</span>
                              <span className="text-blue-400">{irrigationBoost}%</span>
                            </div>
                            <input 
                              type="range" 
                              min="0" max="100" 
                              value={irrigationBoost}
                              onChange={(e) => setIrrigationBoost(parseInt(e.target.value))}
                              className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-blue-500"
                            />
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between text-[10px] font-bold text-gray-500 uppercase">
                              <span>Targeted Fertilizer</span>
                              <span className="text-emerald-400">{fertilizerBoost}%</span>
                            </div>
                            <input 
                              type="range" 
                              min="0" max="100" 
                              value={fertilizerBoost}
                              onChange={(e) => setFertilizerBoost(parseInt(e.target.value))}
                              className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                            />
                          </div>
                        </div>

                        <div className="p-4 bg-blue-500/5 border border-blue-500/20 rounded-xl">
                          <div className="flex items-center justify-between mb-4">
                            <p className="text-xs font-bold text-white uppercase">Simulated Health Recovery</p>
                            <TrendingUp className="w-4 h-4 text-emerald-500" />
                          </div>
                          
                          {(() => {
                            // Logic: Too much irrigation or fertilizer causes toxicity/drowning
                            // Optimal range: 60-80%. Above 80% causes rapid health decline.
                            const getEfficiency = (val: number) => {
                              if (val <= 60) return val / 60; // Linear growth
                              if (val <= 80) return 1.0; // Peak efficiency
                              return 1.0 - ((val - 80) / 20) * 2.5; // Sharp decline to -1.5 at 100%
                            };

                            const iEff = getEfficiency(irrigationBoost);
                            const fEff = getEfficiency(fertilizerBoost);

                            const boost = (result.stressed * iEff * 0.7) + (result.critical * fEff * 0.3);
                            const newHealthy = Math.max(0, Math.min(100, result.healthy + boost));
                            const improvement = newHealthy - result.healthy;
                            
                            // Yield Impact: Yield scales with health improvement
                            const baseYield = parseFloat(result.yield.expected.split(' ')[0]);
                            const yieldMultiplier = 1 + (improvement / 100) * 2.0; // 2.0x sensitivity
                            const simulatedYield = Math.max(0, baseYield * yieldMultiplier).toFixed(1);
                            
                            const isToxic = iEff < 0 || fEff < 0;
                            const isDeclining = improvement < 0;

                            return (
                              <div className="space-y-4">
                                <div className="flex items-end justify-between">
                                  <div>
                                    <p className="text-[10px] text-gray-500 uppercase mb-1">Projected Health</p>
                                    <p className={`text-2xl font-bold ${isDeclining ? 'text-red-500' : 'text-white'}`}>
                                      {newHealthy.toFixed(1)}%
                                    </p>
                                  </div>
                                  <div className="text-right">
                                    <p className={`text-[10px] uppercase font-bold mb-1 ${isDeclining ? 'text-red-500' : 'text-emerald-500'}`}>
                                      {isDeclining ? 'Health Loss' : 'Improvement'}
                                    </p>
                                    <p className={`text-lg font-bold ${isDeclining ? 'text-red-500' : 'text-emerald-500'}`}>
                                      {improvement > 0 ? '+' : ''}{improvement.toFixed(1)}%
                                    </p>
                                  </div>
                                </div>

                                <div className="p-3 bg-black/20 rounded-lg border border-white/5">
                                  <div className="flex items-center justify-between mb-1">
                                    <p className="text-[10px] text-gray-500 uppercase font-bold">Simulated Yield</p>
                                    <span className={`text-xs font-bold ${isDeclining ? 'text-red-400' : 'text-emerald-400'}`}>
                                      {simulatedYield} T/Ha
                                    </span>
                                  </div>
                                  <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                                    <motion.div 
                                      initial={{ width: 0 }}
                                      animate={{ width: `${Math.min(100, (parseFloat(simulatedYield) / 20) * 100)}%` }}
                                      className={`h-full ${isDeclining ? 'bg-red-500' : 'bg-emerald-500'}`}
                                    />
                                  </div>
                                </div>

                                {isToxic && (
                                  <div className="flex items-center gap-2 p-2 bg-red-500/10 border border-red-500/20 rounded-lg">
                                    <AlertTriangle className="w-3 h-3 text-red-500" />
                                    <p className="text-[10px] text-red-400 font-bold uppercase">Warning: Resource Over-Saturation Detected</p>
                                  </div>
                                )}

                                <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                                  <div 
                                    className={`h-full transition-all duration-500 ${isDeclining ? 'bg-red-500' : 'bg-emerald-500'}`}
                                    style={{ width: `${newHealthy}%` }}
                                  />
                                </div>
                              </div>
                            );
                          })()}
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Field Alerts (Agriculture Mode) */}
                <AnimatePresence mode="wait">
                  {result && (
                    <motion.div 
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="bg-white/5 border border-white/10 rounded-2xl p-6"
                    >
                      <h3 className="text-sm font-mono text-orange-500 uppercase tracking-widest mb-6">Field Alerts</h3>
                      <div className="space-y-4">
                        {result.alerts.map((alert) => (
                          <div key={alert.id} className="p-4 bg-white/5 border border-white/10 rounded-xl space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-bold text-orange-500 uppercase">{alert.label}</span>
                              <div className={`w-2 h-2 rounded-full ${alert.severity === 'Critical' ? 'bg-red-600' : alert.severity === 'High' ? 'bg-red-500' : 'bg-orange-500'}`} />
                            </div>
                            <h4 className="text-sm font-bold text-white">{alert.issue}</h4>
                            <p className="text-xs text-gray-400">{alert.reason}</p>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Stats Card */}
                <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
                  <h3 className={`text-sm font-mono uppercase tracking-widest mb-6 ${activeMode === 'agriculture' ? 'text-emerald-500' : 'text-blue-500'}`}>
                    {activeMode === 'agriculture' ? 'Sector Metrics' : 'Site Stability Metrics'}
                  </h3>
                  <div className="space-y-6">
                    <MetricBar 
                      label={activeMode === 'agriculture' ? 'Healthy Growth' : 'Stable Ground'} 
                      value={result?.healthy || 0} 
                      color={activeMode === 'agriculture' ? 'bg-emerald-500' : 'bg-blue-500'} 
                      icon={activeMode === 'agriculture' ? <Leaf className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />} 
                    />
                    <MetricBar 
                      label={activeMode === 'agriculture' ? 'Stress Level' : 'Hazard Potential'} 
                      value={result?.stressed || 0} 
                      color="bg-yellow-500" 
                      icon={<AlertTriangle className="w-4 h-4" />} 
                    />
                    <MetricBar 
                      label={activeMode === 'agriculture' ? 'Critical Area' : 'Active Hazard'} 
                      value={result?.critical || 0} 
                      color="bg-red-500" 
                      icon={activeMode === 'agriculture' ? <Droplets className="w-4 h-4" /> : <Activity className="w-4 h-4" />} 
                    />
                    {activeMode === 'risk' && result && (
                      <>
                        <MetricBar 
                          label="Water Inundation" 
                          value={result.healthy === 0 ? 0 : Math.round((result.metadata.humidity - 30) / 0.7)} 
                          color="bg-blue-500" 
                          icon={<CloudRain className="w-4 h-4" />} 
                        />
                        <MetricBar 
                          label="Thermal Anomaly" 
                          value={Math.min(100, Math.max(0, (result.metadata.temp - 20) * 2))} 
                          color="bg-orange-500" 
                          icon={<Flame className="w-4 h-4" />} 
                        />
                      </>
                    )}
                  </div>
                </div>
              </>
            )}

            {/* Risk Mode Content */}
            {activeMode === 'risk' && (
              <>
                {/* Disaster Risk Analysis */}
                <AnimatePresence mode="wait">
                  {result && (
                    <motion.div 
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="bg-white/5 border border-white/10 rounded-2xl p-6"
                    >
                      <div className="flex flex-col gap-6 mb-6">
                        <div className="flex items-center justify-between">
                          <h3 className="text-sm font-mono text-red-500 uppercase tracking-widest">Disaster Risk Prediction</h3>
                          <div className="animate-pulse bg-red-500/20 px-2 py-1 rounded border border-red-500/30">
                            <span className="text-[10px] font-bold text-red-500 uppercase">Hazard Alert</span>
                          </div>
                        </div>

                        {result.detectedDisasterType && result.detectedDisasterType !== 'None Detected' && (
                          <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-xl flex items-center gap-4">
                            <div className="w-12 h-12 bg-red-500/20 rounded-full flex items-center justify-center border border-red-500/30">
                              {result.detectedDisasterType.includes('Volcanic') && <Flame className="w-6 h-6 text-red-500" />}
                              {result.detectedDisasterType.includes('Flood') && <CloudRain className="w-6 h-6 text-blue-500" />}
                              {result.detectedDisasterType.includes('Seismic') && <Activity className="w-6 h-6 text-orange-500" />}
                              {result.detectedDisasterType.includes('Wildfire') && <Flame className="w-6 h-6 text-orange-600" />}
                            </div>
                            <div>
                              <p className="text-[10px] font-bold text-red-400 uppercase tracking-widest mb-1">Primary Disaster Detected</p>
                              <h4 className="text-lg font-bold text-white leading-none">{result.detectedDisasterType}</h4>
                            </div>
                          </div>
                        )}

                        {/* Filters */}
                        <div className="grid grid-cols-2 gap-3">
                          <div className="space-y-2">
                            <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Filter Type</label>
                            <select 
                              value={riskFilterType}
                              onChange={(e) => setRiskFilterType(e.target.value as any)}
                              className="w-full bg-black/40 border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white focus:outline-none focus:border-red-500/50"
                            >
                              <option value="All">All Types</option>
                              <option value="Flood">Flood</option>
                              <option value="Fire">Fire</option>
                              <option value="Storm">Storm</option>
                              <option value="Erosion">Erosion</option>
                              <option value="Volcanic">Volcanic</option>
                              <option value="Earthquake">Earthquake</option>
                            </select>
                          </div>
                          <div className="space-y-2">
                            <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Filter Severity</label>
                            <select 
                              value={riskFilterSeverity}
                              onChange={(e) => setRiskFilterSeverity(e.target.value as any)}
                              className="w-full bg-black/40 border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white focus:outline-none focus:border-red-500/50"
                            >
                              <option value="All">All Severities</option>
                              <option value="Low">Low</option>
                              <option value="Medium">Medium</option>
                              <option value="High">High</option>
                              <option value="Critical">Critical</option>
                            </select>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-4">
                        {result.disasterRisks
                          .filter(risk => riskFilterType === 'All' || risk.type === riskFilterType)
                          .filter(risk => riskFilterSeverity === 'All' || risk.severity === riskFilterSeverity)
                          .map((risk) => (
                          <div key={risk.id} className="p-4 bg-red-500/5 border border-red-500/20 rounded-xl space-y-3">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                {risk.type === 'Flood' && <CloudRain className="w-4 h-4 text-blue-500" />}
                                {risk.type === 'Fire' && <Sun className="w-4 h-4 text-orange-500" />}
                                {risk.type === 'Storm' && <Wind className="w-4 h-4 text-gray-400" />}
                                {risk.type === 'Erosion' && <Layers className="w-4 h-4 text-amber-700" />}
                                {risk.type === 'Volcanic' && <Flame className="w-4 h-4 text-red-500" />}
                                {risk.type === 'Earthquake' && <Activity className="w-4 h-4 text-orange-500" />}
                                <span className="text-xs font-bold text-white uppercase">{risk.type} Hazard</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <span className={`text-[9px] px-2 py-0.5 rounded-full font-bold uppercase ${
                                  risk.severity === 'Critical' ? 'bg-red-500 text-white' :
                                  risk.severity === 'High' ? 'bg-orange-500 text-white' :
                                  risk.severity === 'Medium' ? 'bg-yellow-500 text-black' :
                                  'bg-blue-500 text-white'
                                }`}>
                                  {risk.severity}
                                </span>
                                <span className="text-[10px] bg-red-500/20 text-red-500 px-2 py-0.5 rounded-full font-bold uppercase border border-red-500/30">{risk.probability}% Prob.</span>
                              </div>
                            </div>
                            <p className="text-xs text-gray-300 font-medium leading-tight">Location: {risk.location}</p>
                            <div className="p-3 bg-black/40 rounded-lg border border-white/5">
                              <p className="text-[11px] text-gray-400 leading-relaxed">
                                <span className="font-bold text-red-500/80">Warning:</span> {risk.warning}
                              </p>
                            </div>
                            <div className="pt-2 border-t border-red-500/10">
                              <p className="text-[11px] text-red-400 font-bold uppercase tracking-wider mb-1 flex items-center gap-1">
                                <Target className="w-3 h-3" />
                                Impact Analysis
                              </p>
                              <p className="text-xs text-gray-300 italic">{risk.impact}</p>
                            </div>
                          </div>
                        ))}
                        {result.disasterRisks
                          .filter(risk => riskFilterType === 'All' || risk.type === riskFilterType)
                          .filter(risk => riskFilterSeverity === 'All' || risk.severity === riskFilterSeverity)
                          .length === 0 && (
                          <div className="p-8 text-center border border-dashed border-white/10 rounded-xl">
                            <p className="text-xs text-gray-500">No hazards match the selected filters.</p>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Alert Areas Prediction Section */}
                <AnimatePresence mode="wait">
                  {result && (
                    <motion.div 
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="bg-white/5 border border-white/10 rounded-2xl p-6"
                    >
                      <h3 className="text-sm font-mono text-red-500 uppercase tracking-widest mb-6">Alert Areas Prediction</h3>
                      <div className="space-y-4">
                        {result.alerts.map((alert) => (
                          <button 
                            key={alert.id}
                            onClick={() => {
                              setSelectedSector(alert);
                              setShowUploadPrompt(true);
                            }}
                            className="w-full text-left p-4 bg-red-500/5 hover:bg-red-500/10 border border-red-500/20 rounded-xl transition-all group"
                          >
                            <div className="flex items-center justify-between mb-3">
                              <span className="text-xs font-bold text-red-500 uppercase">{alert.label}</span>
                              <div className={`w-2 h-2 rounded-full animate-pulse ${alert.severity === 'Critical' ? 'bg-red-600 shadow-[0_0_10px_rgba(220,38,38,0.5)]' : alert.severity === 'High' ? 'bg-red-500' : 'bg-orange-500'}`} />
                            </div>
                            <h4 className="text-lg font-bold text-white mb-1">{alert.issue}</h4>
                            <p className="text-xs text-gray-400 mb-4">{alert.reason}</p>
                            <div className="p-3 bg-black/40 rounded-lg border border-white/5 mb-4">
                              <p className="text-[11px] text-gray-500 leading-relaxed">
                                <span className="font-bold text-gray-400">Analysis:</span> {alert.details}
                              </p>
                            </div>
                            <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-red-500/60 group-hover:text-red-500 transition-colors">
                              <span>Upload Sector Image</span>
                              <Upload className="w-3 h-3" />
                            </div>
                          </button>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Risk Assessment */}
                <AnimatePresence mode="wait">
                  {result && (
                    <motion.div 
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="bg-white/5 border border-white/10 rounded-2xl p-6"
                    >
                      <h3 className="text-sm font-mono text-orange-500 uppercase tracking-widest mb-4">Risk Assessment</h3>
                      <div className="space-y-4">
                        {result.risks.length > 0 ? result.risks.map((risk, idx) => (
                          <div key={idx} className="p-4 bg-orange-500/10 border border-orange-500/20 rounded-xl space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-bold text-orange-500 uppercase">{risk.type}</span>
                              <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                                risk.severity === 'Critical' ? 'bg-red-500 text-white' : 'bg-orange-500 text-white'
                              }`}>
                                {risk.severity}
                              </span>
                            </div>
                            <p className="text-xs text-gray-400 leading-relaxed">{risk.reason}</p>
                          </div>
                        )) : (
                          <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl flex items-center gap-3">
                            <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                            <span className="text-sm text-emerald-500 font-medium">No immediate risks detected.</span>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}

            {/* Recommendations */}
            <AnimatePresence mode="wait">
              {result && (
                <motion.div 
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="bg-white/5 border border-white/10 rounded-2xl p-6"
                >
                  <h3 className={`text-sm font-mono uppercase tracking-widest mb-6 ${activeMode === 'agriculture' ? 'text-emerald-500' : 'text-orange-500'}`}>
                    AI Recommendations
                  </h3>
                  <div className="space-y-3">
                    {result.recommendations.map((rec, idx) => (
                      <div key={idx} className="flex items-start gap-3 p-3 bg-white/5 border border-white/10 rounded-xl group hover:border-white/20 transition-colors">
                        <div className={`mt-1 w-1.5 h-1.5 rounded-full shrink-0 ${activeMode === 'agriculture' ? 'bg-emerald-500' : 'bg-orange-500'}`} />
                        <p className="text-xs text-gray-300 leading-relaxed">{rec}</p>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Explainable AI (XAI) Panel - Visible in both modes */}
            <AnimatePresence mode="wait">
              {result && (
                <motion.div 
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="bg-white/5 border border-white/10 rounded-2xl p-6"
                >
                  <h3 className={`text-sm font-mono uppercase tracking-widest mb-6 ${activeMode === 'agriculture' ? 'text-emerald-500' : 'text-blue-500'}`}>
                    Explainable AI (XAI) Panel
                  </h3>
                  <div className="space-y-3">
                    {activeMode === 'agriculture' ? (
                      <>
                        <div className="p-3 bg-white/5 border border-white/10 rounded-xl">
                          <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">Health Logic</p>
                          <p className="text-xs text-gray-300">
                            🟢 <span className="text-emerald-400 font-bold">{result.healthy}%</span> green pixels detected. 
                            {result.healthy > 70 ? ' Optimal chlorophyll absorption indicates healthy growth.' : ' Sub-optimal greenery suggests potential nutrient or water stress.'}
                          </p>
                        </div>
                        <div className="p-3 bg-white/5 border border-white/10 rounded-xl">
                          <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">Stress Logic</p>
                          <p className="text-xs text-gray-300">
                            🟡 <span className="text-yellow-400 font-bold">{result.stressed}%</span> yellow pixels. 
                            {result.stressed > 15 ? ' High percentage of yellowing indicates significant drought stress.' : ' Low stress signatures detected in primary quadrants.'}
                          </p>
                        </div>
                        <div className="p-3 bg-white/5 border border-white/10 rounded-xl">
                          <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">Critical Logic</p>
                          <p className="text-xs text-gray-300">
                            🔴 <span className="text-red-400 font-bold">{result.critical}%</span> red pixels. 
                            {result.critical > 20 ? ' Significant soil exposure or dead biomass requires immediate attention.' : ' Minimal soil exposure detected across the sector.'}
                          </p>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="p-3 bg-white/5 border border-white/10 rounded-xl">
                          <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">Pixel Classification</p>
                          <p className="text-xs text-gray-300">
                            The AI segments the image into <span className="text-blue-400 font-bold">Water</span>, <span className="text-orange-400 font-bold">Thermal</span>, and <span className="text-gray-400 font-bold">Debris</span> classes based on spectral intensity.
                          </p>
                        </div>
                        <div className="p-3 bg-white/5 border border-white/10 rounded-xl">
                          <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">Hazard Detection</p>
                          <p className="text-xs text-gray-300">
                            {result.detectedDisasterType && result.detectedDisasterType !== 'None Detected' 
                              ? `Detected '${result.detectedDisasterType}' via pattern recognition of anomalous surface features.`
                              : 'Scanning for structural displacement and thermal anomalies across the viewport.'}
                          </p>
                        </div>
                      </>
                    )}
                    {result.detectedDisasterType && result.detectedDisasterType !== 'None Detected' && (
                      <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-xl">
                        <p className="text-[10px] text-red-400 uppercase font-bold mb-1">Decision Logic</p>
                        <p className="text-xs text-red-300">
                          🔍 Triggered <span className="font-bold">'{result.detectedDisasterType}'</span> based on anomalous spectral signatures and pixel density.
                        </p>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Environmental Data */}
            <div className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-6">
              <h3 className="text-sm font-mono text-gray-500 uppercase tracking-widest">Technical Metadata</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-gray-500">
                    <Sun className="w-3 h-3" />
                    <span className="text-[10px] uppercase font-bold tracking-wider">UV Index</span>
                  </div>
                  <p className="text-lg font-bold">
                    {result?.metadata.uvIndex.toFixed(1) || '7.2'} 
                    <span className="text-xs text-orange-500 ml-2">
                      {result ? (result.metadata.uvIndex > 7 ? 'High' : 'Moderate') : 'High'}
                    </span>
                  </p>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-gray-500">
                    <CloudRain className="w-3 h-3" />
                    <span className="text-[10px] uppercase font-bold tracking-wider">Humidity</span>
                  </div>
                  <p className="text-lg font-bold">{result?.metadata.humidity || '42'}%</p>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-gray-500">
                    {activeMode === 'agriculture' ? <Leaf className="w-3 h-3" /> : <Wind className="w-3 h-3" />}
                    <span className="text-[10px] uppercase font-bold tracking-wider">
                      {activeMode === 'agriculture' ? 'LAI (Growth)' : 'Wind Speed'}
                    </span>
                  </div>
                  <p className="text-lg font-bold">
                    {activeMode === 'agriculture' 
                      ? (result?.metadata.lai.toFixed(2) || '2.80')
                      : `${(result?.metadata.lai * 5).toFixed(1)} km/h`}
                  </p>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-gray-500">
                    {activeMode === 'agriculture' ? <Thermometer className="w-3 h-3" /> : <Layers className="w-3 h-3" />}
                    <span className="text-[10px] uppercase font-bold tracking-wider">
                      {activeMode === 'agriculture' ? 'Soil pH' : 'Slope Angle'}
                    </span>
                  </div>
                  <p className="text-lg font-bold">
                    {activeMode === 'agriculture'
                      ? (result?.metadata.soilPh.toFixed(1) || '6.5')
                      : `${(result?.metadata.soilPh * 2).toFixed(1)}°`}
                  </p>
                </div>
              </div>
              <div className="pt-4 border-t border-white/5">
                <div className="flex items-center justify-between text-[10px] text-gray-500 font-mono">
                  <span>{activeMode === 'agriculture' ? 'CHLOROPHYLL:' : 'GROUND SATURATION:'}</span>
                  <span className={activeMode === 'agriculture' ? 'text-emerald-500' : 'text-blue-500'}>
                    {activeMode === 'agriculture' 
                      ? `${result?.metadata.chlorophyll || '42'} mg/m²`
                      : `${result?.metadata.chlorophyll || '42'}%`}
                  </span>
                </div>
              </div>
            </div>

          </div>
        </div>
      </main>

      {/* Full Report Modal */}
      <AnimatePresence>
        {showFullReport && result && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 md:p-10">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowFullReport(false)}
              className="absolute inset-0 bg-black/80 backdrop-blur-md"
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="relative w-full max-w-4xl bg-[#12151a] border border-white/10 rounded-3xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
            >
              <div className="p-6 border-b border-white/5 flex items-center justify-between bg-white/5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-500 rounded-xl flex items-center justify-center">
                    <Database className="text-white w-6 h-6" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">{activeMode === 'agriculture' ? 'Comprehensive Field Report' : 'Disaster Risk Assessment Report'}</h2>
                    <p className="text-xs text-blue-400 font-mono uppercase tracking-widest">Session ID: {activeMode === 'agriculture' ? 'AG-992-X' : 'DR-441-Z'}</p>
                  </div>
                </div>
                <button 
                  onClick={() => setShowFullReport(false)}
                  className="p-2 hover:bg-white/10 rounded-full transition-colors"
                >
                  <AlertTriangle className="w-6 h-6 rotate-45" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-8 space-y-10">
                {/* Executive Summary */}
                <section className="space-y-4">
                  <h3 className="text-lg font-bold flex items-center gap-2 text-white">
                    <Info className="w-5 h-5 text-blue-500" />
                    {activeMode === 'agriculture' ? 'Executive Summary' : 'Hazard Assessment Summary'}
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 bg-white/5 rounded-2xl border border-white/10">
                      <p className="text-xs text-gray-500 uppercase font-bold mb-1">
                        {activeMode === 'agriculture' ? 'Overall Health' : 'Site Stability'}
                      </p>
                      <p className={`text-2xl font-bold ${activeMode === 'agriculture' ? 'text-emerald-500' : 'text-blue-500'}`}>
                        {activeMode === 'agriculture' ? `${result.healthy}%` : `${100 - result.critical}%`}
                      </p>
                    </div>
                    <div className="p-4 bg-white/5 rounded-2xl border border-white/10">
                      <p className="text-xs text-gray-500 uppercase font-bold mb-1">
                        {activeMode === 'agriculture' ? 'Yield Forecast' : 'Infrastructure Integrity'}
                      </p>
                      <p className="text-2xl font-bold text-white">{result.yield.expected}</p>
                    </div>
                    <div className="p-4 bg-white/5 rounded-2xl border border-white/10">
                      <p className="text-xs text-gray-500 uppercase font-bold mb-1">Risk Level</p>
                      <p className={`text-2xl font-bold ${result.critical > 20 ? 'text-red-500' : 'text-orange-500'}`}>
                        {result.critical > 20 ? 'Critical' : 'Moderate'}
                      </p>
                    </div>
                  </div>
                </section>

                {/* Detailed Analysis */}
                <section className="space-y-6">
                  <h3 className="text-lg font-bold text-white">
                    {activeMode === 'agriculture' ? 'Detailed Sector Breakdown' : 'Hazard Zone Breakdown'}
                  </h3>
                  <div className="space-y-4">
                    {result.criticalZones.map((zone, i) => (
                      <div key={i} className="p-6 bg-white/5 border border-white/10 rounded-2xl space-y-4">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-red-500">{zone.location}</h4>
                          <span className="text-[10px] bg-red-500 text-white px-3 py-1 rounded-full font-bold uppercase tracking-widest">{zone.severity}</span>
                        </div>
                        <p className="text-sm text-gray-300 leading-relaxed">{zone.analysis}</p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-white/5">
                          <div>
                            <p className="text-xs font-bold text-gray-500 uppercase mb-2">Observed Indicators</p>
                            <ul className="space-y-1">
                              {zone.reasons.map((r, j) => (
                                <li key={j} className="text-xs text-gray-400 flex items-center gap-2">
                                  <div className="w-1 h-1 rounded-full bg-red-500" />
                                  {r}
                                </li>
                              ))}
                            </ul>
                          </div>
                          <div>
                            <p className="text-xs font-bold text-gray-500 uppercase mb-2">Action Plan</p>
                            <p className="text-xs text-emerald-500 font-medium">{zone.action}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>

                {/* Disaster Risks (Risk Mode Only) */}
                {activeMode === 'risk' && (
                  <section className="space-y-6">
                    <h3 className="text-lg font-bold text-white flex items-center gap-2">
                      <AlertTriangle className="w-5 h-5 text-red-500" />
                      Hazard Risk Assessment
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {result.disasterRisks.map((risk, i) => (
                        <div key={i} className="p-4 bg-white/5 border border-white/10 rounded-2xl flex items-start gap-4">
                          <div className={`p-3 rounded-xl ${
                            risk.type === 'Flood' ? 'bg-blue-500/20 text-blue-500' :
                            risk.type === 'Fire' ? 'bg-red-500/20 text-red-500' :
                            risk.type === 'Storm' ? 'bg-purple-500/20 text-purple-500' :
                            'bg-orange-500/20 text-orange-500'
                          }`}>
                            {risk.type === 'Flood' ? <CloudRain className="w-6 h-6" /> :
                             risk.type === 'Fire' ? <Flame className="w-6 h-6" /> :
                             risk.type === 'Storm' ? <Wind className="w-6 h-6" /> :
                             <AlertTriangle className="w-6 h-6" />}
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                              <h4 className="font-bold text-white">{risk.type} Hazard</h4>
                              <span className={`text-[9px] px-2 py-0.5 rounded-full font-bold uppercase ${
                                risk.severity === 'Critical' ? 'bg-red-500 text-white' :
                                risk.severity === 'High' ? 'bg-orange-500 text-white' :
                                risk.severity === 'Medium' ? 'bg-yellow-500 text-black' :
                                'bg-blue-500 text-white'
                              }`}>
                                {risk.severity}
                              </span>
                            </div>
                            <p className="text-xs text-gray-400 mb-2">Location: {risk.location}</p>
                            <div className="p-3 bg-black/20 rounded-lg border border-white/5">
                              <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">Warning</p>
                              <p className="text-xs text-white">{risk.warning}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                )}

                {/* Recommendations */}
                <section className="space-y-4">
                  <h3 className="text-lg font-bold text-white">
                    {activeMode === 'agriculture' ? 'Strategic Recommendations' : 'Disaster Mitigation Plan'}
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {result.recommendations.map((rec, i) => (
                      <div key={i} className="p-4 bg-blue-500/5 border border-blue-500/10 rounded-xl flex items-start gap-3">
                        <div className="mt-1 w-5 h-5 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0">
                          <span className="text-[10px] font-bold text-blue-500">{i + 1}</span>
                        </div>
                        <p className="text-sm text-gray-400">{rec}</p>
                      </div>
                    ))}
                  </div>
                </section>
              </div>

              <div className="p-6 border-t border-white/5 bg-black/40 flex items-center justify-between">
                <p className="text-[10px] text-gray-500 font-mono">REPORT GENERATED: {new Date().toLocaleString()}</p>
                <div className="flex items-center gap-3">
                  <button 
                    onClick={() => setShowFullReport(false)}
                    className="px-6 py-2 text-sm font-bold text-gray-400 hover:text-white transition-colors"
                  >
                    Close
                  </button>
                  <button className="px-6 py-2 bg-emerald-500 hover:bg-emerald-400 text-white rounded-xl text-sm font-bold transition-all">
                    Export PDF
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Footer Info */}
      <footer className="max-w-7xl mx-auto px-6 py-10 border-t border-white/5">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Info className="w-4 h-4" />
            <span>This is a hackathon demo. All analysis is performed locally using color-segmentation logic.</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-[10px] font-mono text-gray-600">BUILD V1.0.4-BETA</span>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">System Online</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

function MetricBar({ label, value, color, icon }: { label: string, value: number, color: string, icon: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs font-medium">
        <div className="flex items-center gap-2 text-gray-400">
          {icon}
          {label}
        </div>
        <span className="text-white font-bold">{value}%</span>
      </div>
      <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          className={`h-full ${color} rounded-full shadow-[0_0_10px_rgba(0,0,0,0.5)]`}
        />
      </div>
    </div>
  );
}
