import { useEffect, useRef, useState } from 'react';
import { useAnalysisFlow } from './hooks/useAnalysisFlow';
import { Nav } from './components/ui/Nav';
import { Footer } from './components/ui/Footer';
import { Hero } from './components/flow/Hero';
import { UploadSection } from './components/flow/UploadSection';
import { ProcessingSection } from './components/flow/ProcessingSection';
import { FaceSection } from './components/flow/FaceSection';
import { ResultsSection } from './components/flow/ResultsSection';
import { TimelineSection } from './components/flow/TimelineSection';
import { FramesSection } from './components/flow/FramesSection';
import { ReportSection } from './components/flow/ReportSection';
import { EmptyResults } from './components/flow/EmptyResults';
import { LiveCameraSection } from './components/flow/LiveCameraSection';
import { ModeSwitch, type AnalysisMode } from './components/flow/ModeSwitch';

export default function App() {
  const flow = useAnalysisFlow();
  const announcedJob = useRef<string | null>(null);
  const [mode, setMode] = useState<AnalysisMode>('upload');
  const live = mode === 'live';

  // Bring the results into view the first time a job completes.
  useEffect(() => {
    if (live) return;
    if (flow.stage !== 'complete' || !flow.result) return;
    if (announcedJob.current === flow.result.jobId) return;
    announcedJob.current = flow.result.jobId;
    document.getElementById('face')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [flow.stage, flow.result, live]);

  // Keep the processing section in view while the pipeline runs.
  useEffect(() => {
    if (live) return;
    if (flow.stage === 'uploading') {
      document.getElementById('processing')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [flow.stage, live]);

  const result = flow.result;
  // Never strand a running upload analysis by switching the view away from it.
  const busy = flow.stage === 'uploading' || flow.stage === 'processing';

  return (
    <>
      <Nav />

      <main>
        <Hero />

        {/* Upload and Live Camera are two entry points to the same model.
            Unmounting the live section is what releases the camera. */}
        <ModeSwitch mode={mode} onChange={setMode} busy={busy} />

        {live ? (
          <LiveCameraSection />
        ) : (
          <>
            <UploadSection flow={flow} />
            <ProcessingSection flow={flow} />

            {result ? (
              <>
                <FaceSection result={result} />
                <ResultsSection result={result} />
                <TimelineSection result={result} />
                <FramesSection result={result} />
                <ReportSection result={result} onReset={flow.reset} />
              </>
            ) : (
              <EmptyResults />
            )}
          </>
        )}
      </main>

      <Footer />

      <span className="sr-only" role="status" aria-live="polite">
        {live ? '' : flow.statusMessage}
      </span>
    </>
  );
}
