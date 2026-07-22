import { Monitor, Moon, Settings as SettingsIcon, Sun } from 'lucide-react';

import { PageHeader } from '@/components/common/page-header';
import { SectionHeader } from '@/components/common/section-header';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { type Theme, useTheme } from '@/providers/theme-provider';
import { cn } from '@/lib/utils';
import { type Density, useSettingsStore } from '@/store/settings-store';

const THEME_OPTIONS: Array<{ value: Theme; label: string; icon: typeof Sun }> = [
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'dark', label: 'Dark', icon: Moon },
  { value: 'system', label: 'System', icon: Monitor },
];

const DENSITY_OPTIONS: Array<{ value: Density; label: string }> = [
  { value: 'comfortable', label: 'Comfortable' },
  { value: 'compact', label: 'Compact' },
];

const PAGE_SIZES = [10, 25, 50, 100];

/**
 * Settings — a functional preferences page wired to the theme + settings stores.
 * This is foundation state, not a feature: it persists appearance and list
 * preferences the rest of the app reads.
 */
export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const density = useSettingsStore((state) => state.density);
  const setDensity = useSettingsStore((state) => state.setDensity);
  const eventsPageSize = useSettingsStore((state) => state.eventsPageSize);
  const setEventsPageSize = useSettingsStore((state) => state.setEventsPageSize);

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <PageHeader
        icon={SettingsIcon}
        title="Settings"
        description="Personalize the console. Preferences are saved on this device."
      />

      <Card>
        <CardContent className="space-y-6 pt-6">
          <section className="space-y-3">
            <SectionHeader title="Appearance" description="Choose how the interface looks." />
            <div className="flex flex-wrap gap-2" role="group" aria-label="Theme">
              {THEME_OPTIONS.map(({ value, label, icon: Icon }) => (
                <Button
                  key={value}
                  variant={theme === value ? 'default' : 'outline'}
                  size="sm"
                  aria-pressed={theme === value}
                  onClick={() => setTheme(value)}
                >
                  <Icon className="size-4" />
                  {label}
                </Button>
              ))}
            </div>
          </section>

          <Separator />

          <section className="space-y-3">
            <SectionHeader title="Density" description="Row spacing in tables and lists." />
            <div className="flex flex-wrap gap-2" role="group" aria-label="Density">
              {DENSITY_OPTIONS.map(({ value, label }) => (
                <Button
                  key={value}
                  variant={density === value ? 'default' : 'outline'}
                  size="sm"
                  aria-pressed={density === value}
                  onClick={() => setDensity(value)}
                >
                  {label}
                </Button>
              ))}
            </div>
          </section>

          <Separator />

          <section className="space-y-3">
            <SectionHeader
              title="Default page size"
              description="Rows shown per page in event lists."
            />
            <div className="flex flex-wrap gap-2" role="group" aria-label="Default page size">
              {PAGE_SIZES.map((size) => (
                <Button
                  key={size}
                  variant={eventsPageSize === size ? 'default' : 'outline'}
                  size="sm"
                  aria-pressed={eventsPageSize === size}
                  onClick={() => setEventsPageSize(size)}
                  className={cn('tabular-nums')}
                >
                  {size}
                </Button>
              ))}
            </div>
          </section>
        </CardContent>
      </Card>
    </div>
  );
}
