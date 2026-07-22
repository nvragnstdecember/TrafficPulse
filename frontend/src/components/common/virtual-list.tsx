import { useEffect, useRef, useState } from 'react';

import { cn } from '@/lib/utils';

export interface VirtualListProps<T> {
  items: T[];
  /** Fixed row height in pixels. */
  rowHeight: number;
  /** Viewport height in pixels. */
  height: number;
  overscan?: number;
  getKey: (item: T, index: number) => string;
  renderItem: (item: T, index: number) => React.ReactNode;
  /** When set, scroll this row into view if it is outside the viewport (H7E). */
  scrollToIndex?: number | null;
  className?: string;
  /** Accessible label for the list region. */
  'aria-label'?: string;
}

/**
 * A lightweight fixed-row-height virtual list (H7C; keyboard-aware in H7E).
 *
 * Renders only the rows within the viewport (plus overscan), so a long event
 * list stays smooth without pulling in a virtualization dependency. Rows are
 * absolutely positioned inside a spacer sized to the full list height.
 * `scrollToIndex` keeps a keyboard-driven active row visible without disturbing
 * manual scrolling otherwise.
 */
export function VirtualList<T>({
  items,
  rowHeight,
  height,
  overscan = 4,
  getKey,
  renderItem,
  scrollToIndex,
  className,
  'aria-label': ariaLabel,
}: VirtualListProps<T>) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);

  // Bring a requested row into view (only when it would otherwise be clipped).
  useEffect(() => {
    const el = containerRef.current;
    if (el == null || scrollToIndex == null || scrollToIndex < 0) return;
    const top = scrollToIndex * rowHeight;
    const bottom = top + rowHeight;
    if (top < el.scrollTop) {
      el.scrollTop = top;
    } else if (bottom > el.scrollTop + height) {
      el.scrollTop = bottom - height;
    }
  }, [scrollToIndex, rowHeight, height]);

  const totalHeight = items.length * rowHeight;
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const visibleCount = Math.ceil(height / rowHeight) + overscan * 2;
  const end = Math.min(items.length, start + visibleCount);
  const slice = items.slice(start, end);

  return (
    <div
      ref={containerRef}
      role="list"
      aria-label={ariaLabel}
      className={cn('overflow-y-auto', className)}
      style={{ height }}
      onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        {slice.map((item, index) => {
          const absoluteIndex = start + index;
          return (
            <div
              key={getKey(item, absoluteIndex)}
              role="listitem"
              style={{
                position: 'absolute',
                top: absoluteIndex * rowHeight,
                left: 0,
                right: 0,
                height: rowHeight,
              }}
            >
              {renderItem(item, absoluteIndex)}
            </div>
          );
        })}
      </div>
    </div>
  );
}
