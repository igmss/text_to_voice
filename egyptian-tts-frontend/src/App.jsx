import { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Label } from '@/components/ui/label.jsx'
import { 
  Mic, 
  Play, 
  Pause, 
  Download, 
  Settings, 
  Volume2, 
  Radio,
  FileAudio,
  Languages,
  Sparkles,
  BarChart3,
  Clock,
  CheckCircle,
  AlertCircle,
  Upload,
  Trash2,
  Copy,
  Share2,
  Globe,
  Server,
  Zap
} from 'lucide-react'
import './App.css'

// Voice presets configuration
const VOICE_PRESETS = {
  'commercial-energetic': {
    name: 'Commercial - Energetic',
    description: 'High-energy commercial voice over',
    icon: '⚡',
    settings: { speed: 1.1, pitch: 1.05, energy: 1.2 }
  },
  'commercial-warm': {
    name: 'Commercial - Warm', 
    description: 'Warm and friendly commercial voice',
    icon: '🤗',
    settings: { speed: 0.95, pitch: 0.98, energy: 1.0 }
  },
  'educational-clear': {
    name: 'Educational - Clear',
    description: 'Clear and measured educational delivery',
    icon: '📚',
    settings: { speed: 0.9, pitch: 1.0, energy: 0.8 }
  },
  'documentary-authoritative': {
    name: 'Documentary - Authoritative',
    description: 'Authoritative documentary narration',
    icon: '🎬',
    settings: { speed: 0.85, pitch: 0.95, energy: 0.9 }
  },
  'audiobook-natural': {
    name: 'Audiobook - Natural',
    description: 'Natural storytelling voice',
    icon: '📖',
    settings: { speed: 0.9, pitch: 1.0, energy: 0.7 }
  },
  'news-professional': {
    name: 'News - Professional',
    description: 'Professional news delivery',
    icon: '📺',
    settings: { speed: 1.0, pitch: 1.0, energy: 1.0 }
  }
}

// Speaker voices configuration
const SPEAKERS = {
  'default': {
    name: 'Default Egyptian Voice',
    gender: 'Mixed',
    age: 'Adult',
    description: 'Standard Egyptian Arabic voice'
  },
  'male-young': {
    name: 'Ahmed',
    gender: 'Male',
    age: 'Young Adult',
    description: 'Energetic young male voice'
  },
  'female-adult': {
    name: 'Fatima',
    gender: 'Female', 
    age: 'Adult',
    description: 'Professional female voice'
  },
  'male-mature': {
    name: 'Omar',
    gender: 'Male',
    age: 'Mature',
    description: 'Authoritative mature male voice'
  }
}

function App() {
  // State management
  const [text, setText] = useState('')
  const [selectedPreset, setSelectedPreset] = useState('commercial-warm')
  const [selectedSpeaker, setSelectedSpeaker] = useState('default')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedAudio, setGeneratedAudio] = useState(null)
  const [audioMetadata, setAudioMetadata] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [apiUrl, setApiUrl] = useState('https://5001-i4bjzbrfzdbf1lnmbrew1-e38eca77.manusvm.computer')
  const [connectionStatus, setConnectionStatus] = useState('disconnected')
  const [systemInfo, setSystemInfo] = useState(null)
  const [error, setError] = useState(null)
  const [progress, setProgress] = useState(0)
  
  const audioRef = useRef(null)

  // Test API connection
  const testConnection = async () => {
    try {
      setConnectionStatus('testing')
      const response = await fetch(`${apiUrl}/api/health`)
      if (response.ok) {
        const data = await response.json()
        setConnectionStatus('connected')
        setError(null)
        return true
      } else {
        setConnectionStatus('error')
        setError(`API returned ${response.status}: ${response.statusText}`)
        return false
      }
    } catch (err) {
      setConnectionStatus('error')
      setError(`Connection failed: ${err.message}`)
      return false
    }
  }

  // Load system information
  const loadSystemInfo = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/system-info`)
      if (response.ok) {
        const data = await response.json()
        setSystemInfo(data)
      }
    } catch (err) {
      console.error('Failed to load system info:', err)
    }
  }

  // Generate voice over
  const generateVoice = async () => {
    if (!text.trim()) {
      setError('Please enter some text to generate voice')
      return
    }

    if (connectionStatus !== 'connected') {
      setError('Please check API connection first')
      return
    }

    setIsGenerating(true)
    setError(null)
    setProgress(0)

    // Simulate progress
    const progressInterval = setInterval(() => {
      setProgress(prev => Math.min(prev + 10, 90))
    }, 200)

    try {
      const response = await fetch(`${apiUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          speaker_id: selectedSpeaker,
          voice_preset: selectedPreset
        })
      })

      clearInterval(progressInterval)
      setProgress(100)

      if (response.ok) {
        const data = await response.json()
        setGeneratedAudio(`${apiUrl}${data.audio_url}`)
        setAudioMetadata(data.metadata)
        setError(null)
      } else {
        const errorData = await response.json()
        setError(`Generation failed: ${errorData.error || 'Unknown error'}`)
      }
    } catch (err) {
      clearInterval(progressInterval)
      setError(`Network error: ${err.message}`)
    } finally {
      setIsGenerating(false)
      setTimeout(() => setProgress(0), 1000)
    }
  }

  // Audio controls
  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const downloadAudio = () => {
    if (generatedAudio) {
      const a = document.createElement('a')
      a.href = generatedAudio
      a.download = `voice_over_${Date.now()}.wav`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    }
  }

  // Quick test functions
  const quickTests = [
    {
      name: 'Arabic Greeting',
      text: 'مرحبا بكم في نظام تحويل النص إلى كلام المصري',
      preset: 'commercial-warm'
    },
    {
      name: 'English Welcome',
      text: 'Welcome to the Egyptian Arabic Text-to-Speech system',
      preset: 'news-professional'
    },
    {
      name: 'Commercial Ad',
      text: 'اكتشف منتجاتنا الجديدة والمميزة بأفضل الأسعار',
      preset: 'commercial-energetic'
    },
    {
      name: 'Educational Content',
      text: 'في هذا الدرس سوف نتعلم كيفية استخدام التكنولوجيا الحديثة',
      preset: 'educational-clear'
    }
  ]

  const runQuickTest = (test) => {
    setText(test.text)
    setSelectedPreset(test.preset)
    setTimeout(() => generateVoice(), 100)
  }

  // Effects
  useEffect(() => {
    testConnection()
  }, [apiUrl])

  useEffect(() => {
    if (connectionStatus === 'connected') {
      loadSystemInfo()
    }
  }, [connectionStatus])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full">
              <Mic className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Egyptian Arabic TTS
            </h1>
          </div>
          <p className="text-xl text-muted-foreground">
            Professional Voice Over System
          </p>
          
          {/* Connection Status */}
          <div className="flex items-center justify-center gap-2 mt-4">
            {connectionStatus === 'connected' && (
              <Badge variant="default" className="bg-green-500">
                <CheckCircle className="h-3 w-3 mr-1" />
                Connected
              </Badge>
            )}
            {connectionStatus === 'testing' && (
              <Badge variant="secondary">
                <Clock className="h-3 w-3 mr-1" />
                Testing...
              </Badge>
            )}
            {connectionStatus === 'error' && (
              <Badge variant="destructive">
                <AlertCircle className="h-3 w-3 mr-1" />
                Disconnected
              </Badge>
            )}
          </div>
        </div>

        <Tabs defaultValue="generate" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="generate">Generate Voice</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
            <TabsTrigger value="system">System Info</TabsTrigger>
          </TabsList>

          {/* Generate Voice Tab */}
          <TabsContent value="generate" className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-2">
              {/* Text Input */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Languages className="h-5 w-5" />
                    Text Input
                  </CardTitle>
                  <CardDescription>
                    Enter your text in Arabic or English
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    placeholder="مرحبا بكم في نظام تحويل النص إلى كلام المصري&#10;&#10;Hello, welcome to the Egyptian Arabic TTS system"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    className="min-h-[120px] text-lg"
                  />
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="speaker">Speaker Voice</Label>
                      <Select value={selectedSpeaker} onValueChange={setSelectedSpeaker}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.entries(SPEAKERS).map(([id, speaker]) => (
                            <SelectItem key={id} value={id}>
                              {speaker.name} ({speaker.gender}, {speaker.age})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div>
                      <Label htmlFor="preset">Voice Preset</Label>
                      <Select value={selectedPreset} onValueChange={setSelectedPreset}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.entries(VOICE_PRESETS).map(([id, preset]) => (
                            <SelectItem key={id} value={id}>
                              {preset.icon} {preset.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <Button 
                    onClick={generateVoice} 
                    disabled={isGenerating || !text.trim() || connectionStatus !== 'connected'}
                    className="w-full"
                    size="lg"
                  >
                    {isGenerating ? (
                      <>
                        <Radio className="h-4 w-4 mr-2 animate-pulse" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4 mr-2" />
                        Generate Voice Over
                      </>
                    )}
                  </Button>

                  {isGenerating && (
                    <div className="space-y-2">
                      <Progress value={progress} className="w-full" />
                      <p className="text-sm text-muted-foreground text-center">
                        Processing your text...
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Audio Player */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Volume2 className="h-5 w-5" />
                    Generated Audio
                  </CardTitle>
                  <CardDescription>
                    Play and download your voice over
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {generatedAudio ? (
                    <>
                      <audio
                        ref={audioRef}
                        src={generatedAudio}
                        onPlay={() => setIsPlaying(true)}
                        onPause={() => setIsPlaying(false)}
                        onEnded={() => setIsPlaying(false)}
                        className="w-full"
                        controls
                      />
                      
                      <div className="flex gap-2">
                        <Button onClick={togglePlayback} variant="outline" size="sm">
                          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                        </Button>
                        <Button onClick={downloadAudio} variant="outline" size="sm">
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </Button>
                      </div>

                      {audioMetadata && (
                        <div className="grid grid-cols-2 gap-4 p-4 bg-muted rounded-lg">
                          <div className="text-sm">
                            <span className="font-medium">Duration:</span> {audioMetadata.duration?.toFixed(2)}s
                          </div>
                          <div className="text-sm">
                            <span className="font-medium">Quality:</span> {((audioMetadata.quality_score || 0) * 100).toFixed(0)}%
                          </div>
                          <div className="text-sm">
                            <span className="font-medium">Sample Rate:</span> {audioMetadata.sample_rate}Hz
                          </div>
                          <div className="text-sm">
                            <span className="font-medium">Method:</span> {audioMetadata.synthesis_method}
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">
                      <FileAudio className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>No audio generated yet</p>
                      <p className="text-sm">Enter text and click generate to create voice over</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Quick Tests */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Quick Tests
                </CardTitle>
                <CardDescription>
                  Try these sample texts to test different voice styles
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {quickTests.map((test, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      onClick={() => runQuickTest(test)}
                      disabled={isGenerating || connectionStatus !== 'connected'}
                      className="h-auto p-4 text-left"
                    >
                      <div>
                        <div className="font-medium">{test.name}</div>
                        <div className="text-xs text-muted-foreground mt-1">
                          {VOICE_PRESETS[test.preset]?.name}
                        </div>
                      </div>
                    </Button>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Error Display */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  API Configuration
                </CardTitle>
                <CardDescription>
                  Configure your TTS API connection
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="api-url">API Base URL</Label>
                  <Input
                    id="api-url"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    placeholder="https://your-api-url.com"
                  />
                </div>
                
                <Button onClick={testConnection} variant="outline">
                  <Globe className="h-4 w-4 mr-2" />
                  Test Connection
                </Button>
              </CardContent>
            </Card>

            {/* Voice Presets Info */}
            <Card>
              <CardHeader>
                <CardTitle>Available Voice Presets</CardTitle>
                <CardDescription>
                  Choose the right preset for your content type
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  {Object.entries(VOICE_PRESETS).map(([id, preset]) => (
                    <div key={id} className="flex items-center gap-4 p-4 border rounded-lg">
                      <div className="text-2xl">{preset.icon}</div>
                      <div className="flex-1">
                        <h4 className="font-medium">{preset.name}</h4>
                        <p className="text-sm text-muted-foreground">{preset.description}</p>
                      </div>
                      <Badge variant={selectedPreset === id ? "default" : "secondary"}>
                        {selectedPreset === id ? "Selected" : "Available"}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* System Info Tab */}
          <TabsContent value="system" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5" />
                  System Information
                </CardTitle>
                <CardDescription>
                  Current system status and capabilities
                </CardDescription>
              </CardHeader>
              <CardContent>
                {systemInfo ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <h4 className="font-medium">System Details</h4>
                        <div className="text-sm space-y-1">
                          <div><span className="font-medium">System:</span> {systemInfo.system}</div>
                          <div><span className="font-medium">Version:</span> {systemInfo.version}</div>
                          <div><span className="font-medium">Languages:</span> {systemInfo.languages?.join(', ')}</div>
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <h4 className="font-medium">Capabilities</h4>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(systemInfo.capabilities || {}).map(([key, value]) => (
                            <Badge key={key} variant={value ? "default" : "secondary"}>
                              {key.replace(/_/g, ' ')}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <h4 className="font-medium">Available Resources</h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="font-medium">Voice Presets:</span> {systemInfo.voice_presets?.length || 0}
                        </div>
                        <div>
                          <span className="font-medium">Speakers:</span> {systemInfo.speakers?.length || 0}
                        </div>
                        <div>
                          <span className="font-medium">Sample Rates:</span> {systemInfo.sample_rates?.join(', ') || 'N/A'}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>System information not available</p>
                    <p className="text-sm">Check your API connection</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

export default App

