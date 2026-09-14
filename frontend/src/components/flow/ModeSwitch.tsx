import { cx } from '../../lib/format';
import { IconFace, IconUpload } from '../ui/icons';

export type AnalysisMode = 'upload' | 'live';

const OPTIONS: { id: AnalysisMode; label: string; icon: typeof IconUpload }[] = [
  { id: 'upload', label: 'Upload Video', icon: IconUpload },
  { id: 'live', label: 'Live Camera', icon: IconFace },
];

/**
 * The two entry points into the same model. Switching away from Live unmounts
 * the camera section, which is what stops the tracks — so the switch is
 * disabled only while an upload analysis is mid-flight, never while live.
 */
export function ModeSwitch({
  mode,
  onChange,
  busy,
}: {
  mode: AnalysisMode;
  onChange: (mode: AnalysisMode) => void;
  busy: boolean;
}) {
  return (
    <div id="mode" className="w-content-width mx-auto pt-4 scroll-mt-24">
      <div className="flex flex-col items-center gap-2">
        <div
          role="tablist"
          aria-label="Analysis source"
          className="card rounded-token p-1.5 inline-flex gap-1.5 w-full sm:w-auto"
        >
          {OPTIONS.map((option) => {
            const active = mode === option.id;
            const Icon = option.icon;
            return (
              <button
                key={option.id}
                role="tab"
                type="button"
                aria-selected={active}
                disabled={busy && !active}
                onClick={() => onChange(option.id)}
                className={cx(
                  'flex-1 sm:flex-none inline-flex items-center justify-center gap-2',
                  'h-11 px-5 sm:px-7 rounded-token text-sm font-medium whitespace-nowrap',
                  'transition-all duration-300 ease-out cursor-pointer',
                  'disabled:opacity-40 disabled:cursor-not-allowed',
                  active
                    ? 'primary-button text-primary-cta-text'
                    : 'text-foreground/60 hover:text-foreground hover:bg-foreground/5',
                )}
              >
                <Icon className="size-4" />
                {option.label}
              </button>
            );
          })}
        </div>

        {busy && (
          <p className="text-sm text-foreground/45">
            Finish or cancel the running analysis to switch source.
          </p>
        )}
      </div>
    </div>
  );
}
