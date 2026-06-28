/* The Grounded brand mark: a gradient tile with a chat bubble whose tail rests on
 * a "ground" line — grounded conversation. Pure presentational, safe to use in
 * both server and client components. */

export function Logo({ size = 30, withWordmark = false }: { size?: number; withWordmark?: boolean }) {
  const mark = (
    <svg
      className="logo-mark"
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      role="img"
      aria-label="Grounded"
    >
      <defs>
        <linearGradient id="grad-logo" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#6366f1" />
          <stop offset="0.5" stopColor="#8b5cf6" />
          <stop offset="1" stopColor="#d946ef" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="9" fill="url(#grad-logo)" />
      {/* chat bubble */}
      <path
        d="M9 9.5h14a2.5 2.5 0 0 1 2.5 2.5v6a2.5 2.5 0 0 1-2.5 2.5h-7.5L11 23.5V20.5H9A2.5 2.5 0 0 1 6.5 18v-6A2.5 2.5 0 0 1 9 9.5Z"
        fill="#fff"
      />
      {/* three dots = conversation */}
      <circle cx="12" cy="15" r="1.5" fill="#8b5cf6" />
      <circle cx="16" cy="15" r="1.5" fill="#8b5cf6" />
      <circle cx="20" cy="15" r="1.5" fill="#8b5cf6" />
    </svg>
  );

  if (!withWordmark) return mark;
  return (
    <span className="row" style={{ gap: 10 }}>
      {mark}
      <span style={{ fontFamily: "Sora, Inter, sans-serif", fontWeight: 800, fontSize: 18 }}>
        Grounded
      </span>
    </span>
  );
}
