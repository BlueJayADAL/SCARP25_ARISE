import { useEffect, useRef } from 'react';

function supportsEmoji (emoji){
  const ctx = document.createElement("canvas").getContext("2d");
  ctx.canvas.width = ctx.canvas.height = 1;
  ctx.fillText(emoji, -4, 4);
  return ctx.getImageData(0, 0, 1, 1).data[3] > 0; // returns false if transparent pixel (alpha=0)
}

export default function HandCursor({ position, isLeftClicking }) {
  const lastHovered = useRef(null);
  const clickedRef  = useRef(false);
  const cursorEmoji = supportsEmoji("👆") ? "👆" : "^";

  useEffect(() => {
    const { x, y } = position;
    const el = document.elementFromPoint(x, y);
    

    /* ---------- HOVER simulation ---------- */
    if (el !== lastHovered.current) {
      // remove previous hover state
      if (lastHovered.current) {
        lastHovered.current.dispatchEvent(
          new MouseEvent('mouseout', { bubbles: true })
        );
        lastHovered.current.classList.remove('virtual-hover');
      }
      // add hover to new element
      if (el) {
        el.dispatchEvent(
          new MouseEvent('mouseover', { bubbles: true })
        );
        el.classList.add('virtual-hover');
      }
      lastHovered.current = el;
    }

    /* ---------- CLICK simulation ---------- */
    if (isLeftClicking && !clickedRef.current) {
      clickedRef.current = true;
      el?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    }
    if (!isLeftClicking) clickedRef.current = false;
  }, [position, isLeftClicking]);

  return (
    <div
      style={{
        position: 'fixed',
        left: position.x,
        top: position.y,
        transform: 'translate(-50%, -50%)',
        fontSize: '40px',
        pointerEvents: 'none',
        zIndex: 9999,
      }}
    >
      {cursorEmoji}
    </div>
  );
}
