type TypingDotsProps = {
  className?: string;
  inline?: boolean;
};

export function TypingDots({ className = "", inline = false }: TypingDotsProps) {
  const classes = ["typing-dots", inline ? "typing-dots-inline" : "", className]
    .filter(Boolean)
    .join(" ");

  return (
    <span className={classes} aria-hidden="true">
      <span></span>
      <span></span>
      <span></span>
    </span>
  );
}
