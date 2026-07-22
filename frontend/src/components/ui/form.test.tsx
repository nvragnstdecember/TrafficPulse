import { zodResolver } from '@hookform/resolvers/zod';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useForm } from 'react-hook-form';
import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

import { Input } from './input';
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from './form';

const schema = z.object({ name: z.string().min(1, 'Name is required') });
type Values = z.infer<typeof schema>;

function DemoForm({ onValid }: { onValid: (values: Values) => void }) {
  const form = useForm<Values>({ resolver: zodResolver(schema), defaultValues: { name: '' } });
  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onValid)}>
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Name</FormLabel>
              <FormControl>
                <Input {...field} />
              </FormControl>
              <FormDescription>Your display name.</FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        <button type="submit">Save</button>
      </form>
    </Form>
  );
}

describe('Form', () => {
  it('associates the label with the control', () => {
    render(<DemoForm onValid={vi.fn()} />);
    expect(screen.getByLabelText('Name')).toBeInTheDocument();
    expect(screen.getByText('Your display name.')).toBeInTheDocument();
  });

  it('shows a validation error and blocks submit', async () => {
    const onValid = vi.fn();
    const user = userEvent.setup();
    render(<DemoForm onValid={onValid} />);

    await user.click(screen.getByRole('button', { name: 'Save' }));
    expect(await screen.findByText('Name is required')).toBeInTheDocument();
    expect(onValid).not.toHaveBeenCalled();
    expect(screen.getByLabelText('Name')).toHaveAttribute('aria-invalid', 'true');
  });

  it('submits when valid', async () => {
    const onValid = vi.fn();
    const user = userEvent.setup();
    render(<DemoForm onValid={onValid} />);

    await user.type(screen.getByLabelText('Name'), 'Ada');
    await user.click(screen.getByRole('button', { name: 'Save' }));
    expect(onValid).toHaveBeenCalledWith({ name: 'Ada' }, expect.anything());
  });
});
