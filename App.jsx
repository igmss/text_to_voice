import React, { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  Mic, 
  Play, 
  Pause, 
  Download, 
  Settings, 
  Volume2, 
  Waveform,
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
  Share2
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
    settings: { speed: 0.9, pitch: 1.0, energy: 0.9 }
  },
  'documentary-authoritative': {
    name: 'Documentary - Authoritative',
    description: 'Authoritative documentary narration',
    icon: '🎬',
    settings: { speed: 0.85, pitch: 0.95, energy: 1.1 }
  },
  'audiobook-natural': {
    name: 'Audiobook - Natural',
    description: 'Natural storytelling voice',
    icon: '📖',
    settings: { speed: 1.0, pitch: 1.0, energy: 0.95 }
  },
  'news-professional': {
    name: 'News - Professional',
    description: 'Professional news delivery',
    icon: '📺',
    settings: { speed: 1.05, pitch: 1.0, energy: 1.05 }
  }
}

// Speaker voices
const SPEAKERS = {
  'default': { name: 'Default Egyptian Voice', gender: 'neutral', age: 'adult' },
  'male-young': { name: 'Ahmed - Young Male', gender: 'male', age: 'young' },
  'female-adult': { name: 'Fatima - Adult Female', gender: 'female', age: 'adult' },
  'male-mature': { name: 'Omar - Mature Male', gender: 'male', age: 'mature' }
}

function App() {
  // State management
  const [text, setText] = useState('')
  const [selectedPreset, setSelectedPreset] = useState('commercial-warm')
  const [selectedSpeaker, setSelectedSpeaker] = useState('default')
  const [customSettings, setCustomSettings] = useState({
    speed: 1.0,
    pitch: 1.0,
    energy: 1.0
  })
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedAudio, setGeneratedAudio] = useState(null)
  const [audioMetadata, setAudioMetadata] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [qualityScore, setQualityScore] = useState(null)
  const [generationHistory, setGenerationHistory] = useState([])
  const [activeTab, setActiveTab] = useState('generate')

  const audioRef = useRef(null)

  // Sample Egyptian Arabic texts for examples
  const sampleTexts = [
    'مرحبا بكم في برنامجنا الجديد، هنقدملكم محتوى مميز ومفيد',
    'المنتج ده هيغير حياتكم للأحسن، جربوه دلوقتي واستفيدوا من العرض',
    'في الدرس ده هنتعلم مع بعض أساسيات مهمة جداً',
    'كان يا ما كان في قديم الزمان، حكاية جميلة هنحكيهالكم النهاردة'
  ]

  // Simulate audio generation
  const generateVoiceOver = async () => {
    if (!text.trim()) {
      alert('من فضلك أدخل النص أولاً')
      return
    }

    setIsGenerating(true)
    setProgress(0)

    // Simulate generation progress
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval)
          return 90
        }
        return prev + 10
      })
    }, 200)

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Create mock audio data
      const mockAudio = {
        url: 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT',
        duration: text.length * 0.08, // Rough estimate
        sampleRate: 48000
      }

      const metadata = {
        text: text,
        speaker: SPEAKERS[selectedSpeaker].name,
        preset: VOICE_PRESETS[selectedPreset].name,
        duration: mockAudio.duration,
        quality: 0.85 + Math.random() * 0.1, // Random quality score
        timestamp: new Date().toISOString()
      }

      setGeneratedAudio(mockAudio)
      setAudioMetadata(metadata)
      setQualityScore(metadata.quality)
      setProgress(100)

      // Add to history
      setGenerationHistory(prev => [{
        id: Date.now(),
        ...metadata,
        audio: mockAudio
      }, ...prev.slice(0, 9)]) // Keep last 10

    } catch (error) {
      console.error('Generation failed:', error)
      alert('حدث خطأ في توليد الصوت')
    } finally {
      setIsGenerating(false)
      setTimeout(() => setProgress(0), 1000)
    }
  }

  // Apply preset settings
  const applyPreset = (presetKey) => {
    const preset = VOICE_PRESETS[presetKey]
    if (preset) {
      setCustomSettings(preset.settings)
      setSelectedPreset(presetKey)
    }
  }

  // Audio playback controls
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

  // Download audio
  const downloadAudio = () => {
    if (generatedAudio) {
      const link = document.createElement('a')
      link.href = generatedAudio.url
      link.download = `voice_over_${Date.now()}.wav`
      link.click()
    }
  }

  // Copy text to clipboard
  const copyText = () => {
    navigator.clipboard.writeText(text)
    alert('تم نسخ النص')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm dark:bg-gray-900/80 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3 rtl:space-x-reverse">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <Mic className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Egyptian Voice Studio
                </h1>
                <p className="text-sm text-muted-foreground">استوديو الصوت المصري المتقدم</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 rtl:space-x-reverse">
              <Badge variant="secondary" className="hidden sm:flex">
                <Languages className="w-3 h-3 mr-1" />
                Egyptian Arabic
              </Badge>
              <Button variant="outline" size="sm">
                <Settings className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="generate" className="flex items-center space-x-2">
              <Mic className="w-4 h-4" />
              <span>Generate</span>
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center space-x-2">
              <Clock className="w-4 h-4" />
              <span>History</span>
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center space-x-2">
              <BarChart3 className="w-4 h-4" />
              <span>Analytics</span>
            </TabsTrigger>
          </TabsList>

          {/* Generate Tab */}
          <TabsContent value="generate" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Text Input Section */}
              <div className="lg:col-span-2 space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 rtl:space-x-reverse">
                      <FileAudio className="w-5 h-5" />
                      <span>Egyptian Arabic Text</span>
                    </CardTitle>
                    <CardDescription>
                      أدخل النص المصري المراد تحويله إلى صوت
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Textarea
                      placeholder="اكتب النص المصري هنا... مثال: مرحبا بكم في برنامجنا الجديد"
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                      className="min-h-32 text-right"
                      dir="rtl"
                    />
                    <div className="flex flex-wrap gap-2">
                      <span className="text-sm text-muted-foreground">أمثلة سريعة:</span>
                      {sampleTexts.map((sample, index) => (
                        <Button
                          key={index}
                          variant="outline"
                          size="sm"
                          onClick={() => setText(sample)}
                          className="text-xs"
                        >
                          مثال {index + 1}
                        </Button>
                      ))}
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex space-x-2 rtl:space-x-reverse">
                        <Button variant="outline" size="sm" onClick={copyText}>
                          <Copy className="w-4 h-4" />
                        </Button>
                        <Button variant="outline" size="sm">
                          <Upload className="w-4 h-4" />
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => setText('')}>
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {text.length} characters
                      </span>
                    </div>
                  </CardContent>
                </Card>

                {/* Voice Settings */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 rtl:space-x-reverse">
                      <Settings className="w-5 h-5" />
                      <span>Voice Settings</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Voice Presets */}
                    <div className="space-y-3">
                      <label className="text-sm font-medium">Voice Preset</label>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {Object.entries(VOICE_PRESETS).map(([key, preset]) => (
                          <Card
                            key={key}
                            className={`cursor-pointer transition-all hover:shadow-md ${
                              selectedPreset === key ? 'ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-950' : ''
                            }`}
                            onClick={() => applyPreset(key)}
                          >
                            <CardContent className="p-3 text-center">
                              <div className="text-2xl mb-1">{preset.icon}</div>
                              <div className="text-sm font-medium">{preset.name}</div>
                              <div className="text-xs text-muted-foreground">{preset.description}</div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </div>

                    {/* Speaker Selection */}
                    <div className="space-y-3">
                      <label className="text-sm font-medium">Speaker Voice</label>
                      <Select value={selectedSpeaker} onValueChange={setSelectedSpeaker}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.entries(SPEAKERS).map(([key, speaker]) => (
                            <SelectItem key={key} value={key}>
                              <div className="flex items-center space-x-2">
                                <span>{speaker.name}</span>
                                <Badge variant="outline" className="text-xs">
                                  {speaker.gender} • {speaker.age}
                                </Badge>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Custom Controls */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Speed</label>
                        <Slider
                          value={[customSettings.speed]}
                          onValueChange={([value]) => setCustomSettings(prev => ({ ...prev, speed: value }))}
                          min={0.5}
                          max={2.0}
                          step={0.1}
                          className="w-full"
                        />
                        <div className="text-xs text-muted-foreground text-center">
                          {customSettings.speed.toFixed(1)}x
                        </div>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Pitch</label>
                        <Slider
                          value={[customSettings.pitch]}
                          onValueChange={([value]) => setCustomSettings(prev => ({ ...prev, pitch: value }))}
                          min={0.7}
                          max={1.3}
                          step={0.05}
                          className="w-full"
                        />
                        <div className="text-xs text-muted-foreground text-center">
                          {customSettings.pitch.toFixed(2)}
                        </div>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Energy</label>
                        <Slider
                          value={[customSettings.energy]}
                          onValueChange={([value]) => setCustomSettings(prev => ({ ...prev, energy: value }))}
                          min={0.5}
                          max={1.5}
                          step={0.1}
                          className="w-full"
                        />
                        <div className="text-xs text-muted-foreground text-center">
                          {customSettings.energy.toFixed(1)}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Generation & Preview Section */}
              <div className="space-y-6">
                {/* Generate Button */}
                <Card>
                  <CardContent className="p-6">
                    <Button
                      onClick={generateVoiceOver}
                      disabled={isGenerating || !text.trim()}
                      className="w-full h-12 text-lg"
                      size="lg"
                    >
                      {isGenerating ? (
                        <>
                          <Sparkles className="w-5 h-5 mr-2 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Mic className="w-5 h-5 mr-2" />
                          Generate Voice Over
                        </>
                      )}
                    </Button>
                    
                    {isGenerating && (
                      <div className="mt-4 space-y-2">
                        <Progress value={progress} className="w-full" />
                        <p className="text-sm text-muted-foreground text-center">
                          Processing Egyptian Arabic text...
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Audio Preview */}
                {generatedAudio && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2 rtl:space-x-reverse">
                        <Volume2 className="w-5 h-5" />
                        <span>Generated Audio</span>
                        {qualityScore && (
                          <Badge variant={qualityScore > 0.8 ? 'default' : 'secondary'}>
                            Quality: {(qualityScore * 100).toFixed(0)}%
                          </Badge>
                        )}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {/* Audio Controls */}
                      <div className="flex items-center space-x-3 rtl:space-x-reverse">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={togglePlayback}
                        >
                          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={downloadAudio}
                        >
                          <Download className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                        >
                          <Share2 className="w-4 h-4" />
                        </Button>
                      </div>

                      {/* Waveform Visualization */}
                      <div className="h-20 bg-muted rounded-lg flex items-center justify-center">
                        <Waveform className="w-8 h-8 text-muted-foreground" />
                        <span className="ml-2 text-sm text-muted-foreground">Audio Waveform</span>
                      </div>

                      {/* Audio Metadata */}
                      {audioMetadata && (
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">Duration:</span>
                            <span className="ml-1 font-medium">{audioMetadata.duration.toFixed(1)}s</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Speaker:</span>
                            <span className="ml-1 font-medium">{audioMetadata.speaker}</span>
                          </div>
                          <div className="col-span-2">
                            <span className="text-muted-foreground">Preset:</span>
                            <span className="ml-1 font-medium">{audioMetadata.preset}</span>
                          </div>
                        </div>
                      )}

                      {/* Hidden audio element */}
                      <audio
                        ref={audioRef}
                        src={generatedAudio.url}
                        onEnded={() => setIsPlaying(false)}
                        className="hidden"
                      />
                    </CardContent>
                  </Card>
                )}

                {/* Quality Assessment */}
                {qualityScore && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2 rtl:space-x-reverse">
                        <CheckCircle className="w-5 h-5" />
                        <span>Quality Assessment</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Overall Quality</span>
                          <Badge variant={qualityScore > 0.8 ? 'default' : 'secondary'}>
                            {(qualityScore * 100).toFixed(0)}%
                          </Badge>
                        </div>
                        <Progress value={qualityScore * 100} className="w-full" />
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div className="flex items-center space-x-1">
                            <CheckCircle className="w-3 h-3 text-green-500" />
                            <span>Voice Over Standards</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <CheckCircle className="w-3 h-3 text-green-500" />
                            <span>Egyptian Dialect</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <CheckCircle className="w-3 h-3 text-green-500" />
                            <span>Audio Quality</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <AlertCircle className="w-3 h-3 text-yellow-500" />
                            <span>Naturalness</span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Generation History</CardTitle>
                <CardDescription>Your recent voice over generations</CardDescription>
              </CardHeader>
              <CardContent>
                {generationHistory.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Clock className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No generations yet. Create your first voice over!</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {generationHistory.map((item) => (
                      <Card key={item.id} className="p-4">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <p className="text-sm font-medium mb-1" dir="rtl">
                              {item.text.substring(0, 100)}...
                            </p>
                            <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                              <span>{item.speaker}</span>
                              <span>{item.preset}</span>
                              <span>{item.duration.toFixed(1)}s</span>
                              <Badge variant="outline">
                                {(item.quality * 100).toFixed(0)}%
                              </Badge>
                            </div>
                          </div>
                          <div className="flex space-x-2">
                            <Button variant="outline" size="sm">
                              <Play className="w-3 h-3" />
                            </Button>
                            <Button variant="outline" size="sm">
                              <Download className="w-3 h-3" />
                            </Button>
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Total Generations</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">{generationHistory.length}</div>
                  <p className="text-sm text-muted-foreground">Voice overs created</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Average Quality</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">
                    {generationHistory.length > 0
                      ? (generationHistory.reduce((acc, item) => acc + item.quality, 0) / generationHistory.length * 100).toFixed(0)
                      : 0}%
                  </div>
                  <p className="text-sm text-muted-foreground">Quality score</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Total Duration</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">
                    {generationHistory.reduce((acc, item) => acc + item.duration, 0).toFixed(1)}s
                  </div>
                  <p className="text-sm text-muted-foreground">Audio generated</p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t bg-white/80 backdrop-blur-sm dark:bg-gray-900/80 mt-12">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              © 2024 Egyptian Voice Studio. Advanced TTS for Egyptian Arabic.
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline">v1.0.0</Badge>
              <Badge variant="outline">48kHz Quality</Badge>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App

