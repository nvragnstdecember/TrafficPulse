import { useState } from 'react';

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
  className?: string;
}

/**
 * A lightweight fixed-row-height virtual list (H7C).
 *
 * Renders only the rows within the viewport (plus overscan), so a long event
 * list stays smooth without pulling in a virtualization dependency. Rows are
 * absolutely positioned inside a spacer sized to the full list height.
 */
export function VirtualList<T>({
  items,
  rowHeight,
  height,
  overscan = 4,
  getKey,
  renderItem,
  className,
}: VirtualListProps<T>) {
  const [scrollTop, setScrollTop] = useState(0);

  const totalHeight = items.length * rowHeight;
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const visibleCount = Math.ceil(height / rowHeight) + overscan * 2;
  const end = Math.min(items.length, start + visibleCount);
  const slice = items.slice(start, end);

  return (
    <div
      role="list"
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
