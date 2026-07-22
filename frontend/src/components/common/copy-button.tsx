import { Check, Copy } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

import { copyText } from '@/lib/clipboard';
import { cn } from '@/lib/utils';

import { Button, type ButtonProps } from '../ui/button';

export interface CopyButtonProps extends Omit<ButtonProps, 'onClick' | 'children'> {
  /** The value copied to the clipboard. */
  value: string;
  /** Accessible label (e.g. "Copy event ID"). */
  label: string;
  /** Optional visible text next to the icon. */
  children?: React.ReactNode;
}

/**
 * A copy-to-clipboard button with transient success feedback (H7E).
 *
 * Wraps the guarded {@link copyText} helper, swaps to a check for ~1.2s on
 * success, and announces the state to assistive tech via a live region — so
 * copying an event/evidence id is one accessible action, reused everywhere.
 */
export function CopyButton({
  value,
  label,
  children,
  variant = 'ghost',
  size = children ? 'sm' : 'icon',
  className,
  ...props
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const onCopy = useCallback(async () => {
    const ok = await copyText(value);
    if (!ok) return;
    setCopied(true);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setCopied(false), 1200);
  }, [value]);

  useEffect(() => () => void (timerRef.current && clearTimeout(timerRef.current)), []);

  return (
    <Button
      type="button"
      variant={variant}
      size={size}
      onClick={onCopy}
      aria-label={copied ? `${label} (copied)` : label}
      className={cn(className)}
      {...props}
    >
      {copied ? (
        <Check className="size-4 text-success" aria-hidden="true" />
      ) : (
        <Copy className="size-4" aria-hidden="true" />
      )}
      {children}
      <span aria-live="polite" className="sr-only">
        {copied ? 'Copied to clipboard' : ''}
      </span>
    </Button>
  );
}
