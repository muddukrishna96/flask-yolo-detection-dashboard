// Defensive no-op handlers for decorative UI elements
// Prevents clicks on placeholder anchors (href="#") from causing navigation or errors.
// Preserves behavior for elements managed by AdminLTE/plugins by skipping elements
// that have data-widget, data-toggle, data-card-widget, data-slide attributes.

(function(){
  'use strict';

  function shouldIgnore(el){
    if(!el) return true;
    // elements that are part of plugin behaviours should be ignored
    var attrs = ['data-widget','data-toggle','data-card-widget','data-slide','data-action','data-target'];
    for(var i=0;i<attrs.length;i++){
      if(el.hasAttribute && el.hasAttribute(attrs[i])) return true;
    }
    // if the element has an inline onclick handler, don't override
    if(el.getAttribute && el.getAttribute('onclick')) return true;
    return false;
  }

  document.addEventListener('click', function(ev){
    try{
      var t = ev.target;
      // find closest anchor or button
      var el = t.closest ? t.closest('a,button') : (t.tagName && (t.tagName.toLowerCase()==='a' || t.tagName.toLowerCase()==='button') ? t : null);
      if(!el) return;

      // Normalize href
      var href = el.getAttribute && el.getAttribute('href');
      // Only target pure placeholders: href="#" or href="" or javascript:void(0)
      var isPlaceholderHref = href === '#' || href === '' || href === 'javascript:void(0)';

      // Also consider anchors without href but meant as placeholders
      var isAnchorNoHref = el.tagName && el.tagName.toLowerCase() === 'a' && !href;

      if((isPlaceholderHref || isAnchorNoHref) && !shouldIgnore(el)){
        // Prevent navigation and stop further processing for this click
        ev.preventDefault();
        ev.stopPropagation();
        // Provide a harmless console message for debugging
        if(window.console && console.info){
          console.info('noop-ui: intercepted placeholder click on', el);
        }
        // Show a small toast informing the user the feature isn't implemented yet
        try{
          showNoopToast('Feature not implemented yet');
        }catch(e){}
      }
    }catch(e){
      // Never throw from this defensive script
      try{ console.warn('noop-ui error', e); }catch(ee){}
    }
  }, true);

})();

// Simple toast implementation (no external deps).
function showNoopToast(message, timeout){
  timeout = timeout || 2500;
  try{
    var containerId = 'noop-toast-container';
    var container = document.getElementById(containerId);
    if(!container){
      container = document.createElement('div');
      container.id = containerId;
      container.setAttribute('aria-live','polite');
      container.style.position = 'fixed';
      container.style.right = '18px';
      container.style.top = '18px';
      container.style.zIndex = 2147483647; /* very high */
      container.style.display = 'flex';
      container.style.flexDirection = 'column';
      container.style.gap = '8px';
      document.body.appendChild(container);
    }

    var toast = document.createElement('div');
    toast.className = 'noop-toast';
    toast.textContent = message;
    toast.style.background = 'rgba(0,0,0,0.82)';
    toast.style.color = '#fff';
    toast.style.padding = '10px 14px';
    toast.style.borderRadius = '8px';
    toast.style.boxShadow = '0 8px 22px rgba(0,0,0,0.28)';
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(-6px)';
    toast.style.transition = 'opacity 220ms ease, transform 220ms ease';
    toast.style.fontSize = '13px';
    toast.style.maxWidth = '320px';
    toast.style.pointerEvents = 'auto';

    container.appendChild(toast);

    // Force reflow then animate in
    window.getComputedStyle(toast).opacity;
    toast.style.opacity = '1';
    toast.style.transform = 'translateY(0)';

    // Remove after timeout
    setTimeout(function(){
      try{
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-6px)';
        setTimeout(function(){ if(toast && toast.parentNode) toast.parentNode.removeChild(toast); }, 260);
      }catch(e){}
    }, timeout);
  }catch(e){ try{ console.warn('noop-ui toast failed', e); }catch(ee){} }
}
