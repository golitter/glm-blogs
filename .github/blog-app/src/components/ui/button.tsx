import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap border-2 border-ink font-black uppercase transition-[background-color,color,box-shadow,transform] duration-150 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-primary/30 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-white shadow-[3px_3px_0_#111] hover:bg-primary-dark hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[2px_2px_0_#111] active:translate-x-[3px] active:translate-y-[3px] active:shadow-none",
        ghost: "bg-sun text-ink shadow-[3px_3px_0_#111] hover:bg-recess hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[2px_2px_0_#111]",
        outline:
          "bg-surface text-ink shadow-[3px_3px_0_#111] hover:bg-recess hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[2px_2px_0_#111]",
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
