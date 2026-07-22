import { CloudUpload } from 'lucide-react';
import { useRef, useState } from 'react';

import { formatBytes } from '@/lib/format';
import { acceptAttribute, uploadConstraints, validateUploadFile } from '@/lib/upload';
import { cn } from '@/lib/utils';

import { buttonVariants } from '../ui/button';

export interface UploadDropzoneProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
  /** An error raised by the workflow (upload failure), shown beneath the zone. */
  error?: string | null;
}

/**
 * The upload surface (H7C): drag-and-drop or browse, with immediate client-side
 * validation.
 *
 * Constraints come from `lib/upload` (configuration), never hardcoded here, and
 * a rejected file is reported inline without touching the network. The zone is a
 * real button, so keyboard and screen-reader users get the same affordance.
 */
export function UploadDropzone({ onFileSelected, disabled = false, error }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [rejection, setRejection] = useState<string | null>(null);

  function handleFiles(files: FileList | null): void {
    const file = files?.[0];
    if (!file) return;
    const validation = validateUploadFile(file);
    if (!validation.ok) {
      setRejection(validation.message);
      return;
    }
    setRejection(null);
    onFileSelected(file);
  }

  const message = rejection ?? error ?? null;

  return (
    <div className="space-y-2">
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          if (!disabled) handleFiles(event.dataTransfer?.files ?? null);
        }}
        aria-label="Upload a video"
        aria-describedby="upload-constraints"
        className={cn(
          'flex w-full flex-col items-center justify-center gap-3 rounded-lg border border-dashed p-10 text-center transition-colors',
          disabled ? 'cursor-not-allowed opacity-60' : 'hover:border-primary/60 hover:bg-accent/40',
          isDragging ? 'border-primary bg-accent' : 'border-border',
        )}
      >
        <span className="flex size-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
          <CloudUpload className="size-6" aria-hidden="true" />
        </span>
        <span className="space-y-1">
          <span className="block text-sm font-medium">Drag a video here, or click to browse</span>
          <span id="upload-constraints" className="block text-sm text-muted-foreground">
            {uploadConstraints.acceptedExtensions.join(', ')} · up to{' '}
            {formatBytes(uploadConstraints.maxBytes)}
          </span>
        </span>
        {/* Styled as a button, but rendered as a span: the whole zone is the button. */}
        <span className={cn(buttonVariants({ variant: 'outline', size: 'sm' }))} aria-hidden="true">
          Choose file
        </span>
      </button>

      <input
        ref={inputRef}
        type="file"
        accept={acceptAttribute()}
        className="sr-only"
        aria-hidden="true"
        tabIndex={-1}
        data-testid="upload-input"
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = '';
        }}
      />

      {message ? (
        <p role="alert" className="text-sm text-destructive">
          {message}
        </p>
      ) : null}
    </div>
  );
}
