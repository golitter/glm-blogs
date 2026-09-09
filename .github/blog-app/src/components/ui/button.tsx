import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "rounded-full inline-flex items-center justify-center gap-2 whitespace-nowrap border border-line font-semibold  transition-[background-color,color,box-shadow,transform] duration-150 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-primary/30 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-white shadow-soft hover:bg-primary-dark hover:-translate-y-0.5  hover:shadow-soft active:scale-[0.98]  active:shadow-none",
        ghost: "bg-recess text-ink shadow-soft hover:bg-recess hover:-translate-y-0.5  hover:shadow-soft",
        outline:
          "bg-surface text-ink shadow-soft hover:bg-recess hover:-translate-y-0.5  hover:shadow-soft",
      },
      size: {
        default: "h-9 px-4 text-sm",
        sm: "h-8 px-3 text-[11px]",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
  };

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
