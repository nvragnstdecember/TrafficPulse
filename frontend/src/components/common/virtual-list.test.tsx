import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { renderWithProviders } from '@/test/utils';

import { VirtualList } from './virtual-list';

const items = Array.from({ length: 500 }, (_, index) => ({ id: `row-${index}` }));

function renderList() {
  return renderWithProviders(
    <VirtualList
      items={items}
      rowHeight={50}
      height={200}
      overscan={2}
      getKey={(item) => item.id}
      renderItem={(item) => <span>{item.id}</span>}
    />,
  );
}

describe('VirtualList', () => {
  it('renders only the visible window, not the whole list', () => {
    renderList();
    expect(screen.getAllByRole('listitem').length).toBeLessThan(20);
    expect(screen.getByText('row-0')).toBeInTheDocument();
    expect(screen.queryByText('row-400')).not.toBeInTheDocument();
  });

  it('sizes the spacer to the full list height', () => {
    renderList();
    const spacer = screen.getByRole('list').firstElementChild as HTMLElement;
    expect(spacer.style.height).toBe('25000px');
  });

  it('renders the rows for the scrolled window', () => {
    renderList();
    const list = screen.getByRole('list');
    fireEvent.scroll(list, { target: { scrollTop: 5000 } });

    expect(screen.getByText('row-100')).toBeInTheDocument();
    expect(screen.queryByText('row-0')).not.toBeInTheDocument();
  });

  it('positions each row absolutely at its index offset', () => {
    renderList();
    const first = screen.getAllByRole('listitem')[0];
    expect(first.style.top).toBe('0px');
    expect(first.style.height).toBe('50px');
  });

  it('renders nothing for an empty list', () => {
    renderWithProviders(
      <VirtualList
        items={[]}
        rowHeight={50}
        height={200}
        getKey={(_item, index) => String(index)}
        renderItem={() => <span>never</span>}
      />,
    );
    expect(screen.queryAllByRole('listitem')).toHaveLength(0);
  });
});
