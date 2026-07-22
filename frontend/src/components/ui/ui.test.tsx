import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { Badge } from './badge';
import { Button } from './button';
import { Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from './dialog';
import { Skeleton } from './skeleton';
import { Spinner } from './spinner';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './tabs';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './tooltip';

describe('Button', () => {
  it('defaults to type="button" and handles clicks', async () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Click</Button>);
    const button = screen.getByRole('button', { name: 'Click' });
    expect(button).toHaveAttribute('type', 'button');
    await userEvent.click(button);
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('applies variant classes', () => {
    render(<Button variant="destructive">Delete</Button>);
    expect(screen.getByRole('button', { name: 'Delete' })).toHaveClass('bg-destructive');
  });

  it('renders as a child element with asChild', () => {
    render(
      <Button asChild>
        <a href="/somewhere">Link</a>
      </Button>,
    );
    const link = screen.getByRole('link', { name: 'Link' });
    expect(link).toHaveAttribute('href', '/somewhere');
  });
});

describe('Badge', () => {
  it('renders its content', () => {
    render(<Badge variant="success">Active</Badge>);
    expect(screen.getByText('Active')).toBeInTheDocument();
  });
});

describe('Spinner / Skeleton', () => {
  it('spinner exposes an accessible status', () => {
    render(<Spinner label="Loading data" />);
    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.getByText('Loading data')).toBeInTheDocument();
  });

  it('skeleton is decorative', () => {
    const { container } = render(<Skeleton className="h-4 w-10" />);
    expect(container.firstChild).toHaveAttribute('aria-hidden', 'true');
  });
});

describe('Dialog', () => {
  it('opens and closes via triggers', async () => {
    const user = userEvent.setup();
    render(
      <Dialog>
        <DialogTrigger>Open</DialogTrigger>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
          <DialogDescription>Body</DialogDescription>
        </DialogContent>
      </Dialog>,
    );

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Open' }));
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('Body')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Close' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
});

describe('Tabs', () => {
  it('switches panels', async () => {
    const user = userEvent.setup();
    render(
      <Tabs defaultValue="a">
        <TabsList>
          <TabsTrigger value="a">Tab A</TabsTrigger>
          <TabsTrigger value="b">Tab B</TabsTrigger>
        </TabsList>
        <TabsContent value="a">Panel A</TabsContent>
        <TabsContent value="b">Panel B</TabsContent>
      </Tabs>,
    );

    expect(screen.getByText('Panel A')).toBeInTheDocument();
    await user.click(screen.getByRole('tab', { name: 'Tab B' }));
    expect(screen.getByText('Panel B')).toBeInTheDocument();
  });
});

describe('Tooltip', () => {
  it('renders content when open', () => {
    render(
      <TooltipProvider>
        <Tooltip open>
          <TooltipTrigger>Hover</TooltipTrigger>
          <TooltipContent>Helpful hint</TooltipContent>
        </Tooltip>
      </TooltipProvider>,
    );
    expect(screen.getAllByText('Helpful hint').length).toBeGreaterThan(0);
  });
});
