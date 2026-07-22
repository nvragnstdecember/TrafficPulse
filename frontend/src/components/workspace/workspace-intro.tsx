import { type LucideIcon, ScanSearch, ShieldCheck, Upload } from 'lucide-react';

interface IntroStep {
  icon: LucideIcon;
  title: string;
  text: string;
}

const STEPS: IntroStep[] = [
  {
    icon: Upload,
    title: '1 · Upload footage',
    text: 'Drop a fixed-camera clip. It is processed locally — nothing leaves this machine.',
  },
  {
    icon: ScanSearch,
    title: '2 · Detection runs',
    text: 'The engine tracks each vehicle and reasons over time — never a single-frame guess.',
  },
  {
    icon: ShieldCheck,
    title: '3 · Review evidence',
    text: 'Inspect every confirmed violation frame by frame, with its rule trace and manifest.',
  },
];

/**
 * A compact, first-run "how it works" hint shown beneath the upload dropzone
 * (H8, demo readiness).
 *
 * Explains the upload → detect → review flow so the app is understandable
 * without a walkthrough. Purely informational — it disappears once a video is
 * in the workspace, and it changes no existing layout.
 */
export function WorkspaceIntro() {
  return (
    <section aria-label="How it works" className="grid gap-3 sm:grid-cols-3">
      {STEPS.map((step) => {
        const Icon = step.icon;
        return (
          <div key={step.title} className="rounded-lg border bg-card p-4">
            <span className="flex size-9 items-center justify-center rounded-md bg-muted text-muted-foreground">
              <Icon className="size-5" aria-hidden="true" />
            </span>
            <h3 className="mt-3 text-sm font-medium">{step.title}</h3>
            <p className="mt-1 text-sm text-muted-foreground">{step.text}</p>
          </div>
        );
      })}
    </section>
  );
}
