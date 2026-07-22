import { Component, type ErrorInfo, type ReactNode } from 'react';

import { FullPageError } from './full-page-error';

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Custom fallback; receives the error and a reset callback. */
  fallback?: (args: { error: Error; reset: () => void }) => ReactNode;
  /** Optional error sink (telemetry). */
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/**
 * App-wide error boundary (H7B).
 *
 * Catches render-time exceptions anywhere below it and shows a recoverable
 * fallback (default: a full-page error with a "Try again" reset) instead of a
 * blank screen. `reset` clears the error so a transient failure can recover
 * without a full reload.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  override state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    this.props.onError?.(error, info);
  }

  reset = (): void => {
    this.setState({ error: null });
  };

  override render(): ReactNode {
    const { error } = this.state;
    if (error) {
      if (this.props.fallback) return this.props.fallback({ error, reset: this.reset });
      return <FullPageError message={error.message} onRetry={this.reset} />;
    }
    return this.props.children;
  }
}
