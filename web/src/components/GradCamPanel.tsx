import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface GradCamPanelProps {
  originalUrl: string
  gradcamBase64: string
  gradcamClass: string
}

export function GradCamPanel({ originalUrl, gradcamBase64, gradcamClass }: GradCamPanelProps) {
  return (
    <Card className="border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl shadow-black/40">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-white/90">
          grad-cam
          <Badge variant="outline" className="border-white/20 text-white/70">
            {gradcamClass}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-2">
            <span className="text-xs text-white/40">original</span>
            <img
              src={originalUrl}
              alt="original x-ray"
              className="aspect-square w-full rounded-lg object-cover"
            />
          </div>
          <div className="flex flex-col gap-2">
            <span className="text-xs text-white/40">heatmap</span>
            <img
              src={`data:image/png;base64,${gradcamBase64}`}
              alt="gradcam heatmap"
              className="aspect-square w-full rounded-lg object-cover"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
