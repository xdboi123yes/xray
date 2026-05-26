/**
 * @file LoadingSkeleton.tsx
 * @description Reusable skeleton placeholder for in-flight async data fetches.
 */

import React from 'react';

interface LoadingSkeletonProps {
  rows?: number;
  className?: string;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({ rows = 3, className = '' }) => (
  <div className={`space-y-3 ${className}`} aria-hidden="true">
    {Array.from({ length: rows }).map((_, i) => (
      <div
        key={i}
        className="h-4 bg-slate-200/60 dark:bg-slate-800/40 rounded-md animate-pulse"
        style={{ width: `${60 + ((i * 17) % 35)}%` }}
      />
    ))}
  </div>
);
