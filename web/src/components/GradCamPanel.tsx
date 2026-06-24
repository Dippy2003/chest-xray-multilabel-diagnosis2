import { motion } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface GradCamPanelProps {
  originalUrl: string
  gradcamBase64: string
  gradcamClass: string
}

export function GradCamPanel({ originalUrl, gradcamBase64, gradcamClass }: GradCamPanelProps) {
  return (
    <Card className="border-white/10 bg-white/4 backdrop-blur-xl shadow-2xl shadow-black/40">
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
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
            className="flex flex-col gap-2"
          >
            <span className="text-xs text-white/40">original</span>
            <div className="overflow-hidden rounded-lg">
              <img
                src={originalUrl}
                alt="original x-ray"
                className="aspect-square w-full object-cover transition-transform duration-500 hover:scale-110"
              />
            </div>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.15 }}
            className="flex flex-col gap-2"
          >
            <span className="text-xs text-white/40">heatmap</span>
            <div className="overflow-hidden rounded-lg">
              <img
                src={`data:image/png;base64,${gradcamBase64}`}
                alt="gradcam heatmap"
                className="aspect-square w-full object-cover transition-transform duration-500 hover:scale-110"
              />
            </div>
          </motion.div>
        </div>
      </CardContent>
    </Card>
  )
}
