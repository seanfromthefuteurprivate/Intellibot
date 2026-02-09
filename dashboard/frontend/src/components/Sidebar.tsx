import React, { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import {
  LayoutDashboard,
  Briefcase,
  ArrowLeftRight,
  Signal,
  CalendarClock,
  TrendingUp,
  ShieldAlert,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
} from 'lucide-react';

interface NavItem {
  name: string;
  path: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  { name: 'Dashboard', path: '/', icon: <LayoutDashboard size={20} /> },
  { name: 'Positions', path: '/positions', icon: <Briefcase size={20} /> },
  { name: 'Trades', path: '/trades', icon: <ArrowLeftRight size={20} /> },
  { name: 'Signals', path: '/signals', icon: <Signal size={20} /> },
  { name: 'Events', path: '/events', icon: <CalendarClock size={20} /> },
  { name: 'Market', path: '/market', icon: <TrendingUp size={20} /> },
  { name: 'Risk', path: '/risk', icon: <ShieldAlert size={20} /> },
];

interface SidebarProps {
  className?: string;
}

const Sidebar: React.FC<SidebarProps> = ({ className = '' }) => {
  const location = useLocation();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  const isActive = (path: string): boolean => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  const toggleCollapse = (): void => {
    setIsCollapsed(!isCollapsed);
  };

  const toggleMobile = (): void => {
    setIsMobileOpen(!isMobileOpen);
  };

  const closeMobile = (): void => {
    setIsMobileOpen(false);
  };

  return (
    <>
      {/* Mobile menu button */}
      <button
        onClick={toggleMobile}
        className="fixed top-4 left-4 z-50 p-2 rounded-lg bg-gray-800 text-gray-200 hover:bg-gray-700 transition-colors md:hidden"
        aria-label="Toggle menu"
      >
        {isMobileOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Mobile overlay */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 md:hidden"
          onClick={closeMobile}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed top-0 left-0 h-full z-40
          bg-gray-900 border-r border-gray-800
          transition-all duration-300 ease-in-out
          ${isCollapsed ? 'w-16' : 'w-64'}
          ${isMobileOpen ? 'translate-x-0' : '-translate-x-full'}
          md:translate-x-0 md:static
          flex flex-col
          ${className}
        `}
      >
        {/* Logo/Title */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-800">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center flex-shrink-0">
              <TrendingUp size={18} className="text-white" />
            </div>
            {!isCollapsed && (
              <span className="text-lg font-bold text-white whitespace-nowrap">
                IntelliBot
              </span>
            )}
          </div>
          {/* Collapse button - desktop only */}
          <button
            onClick={toggleCollapse}
            className="hidden md:flex p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
            aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {isCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-4 px-2 overflow-y-auto">
          <ul className="space-y-1">
            {navItems.map((item) => (
              <li key={item.path}>
                <Link
                  to={item.path}
                  onClick={closeMobile}
                  className={`
                    flex items-center gap-3 px-3 py-2.5 rounded-lg
                    transition-all duration-200
                    group relative
                    ${
                      isActive(item.path)
                        ? 'bg-emerald-600/20 text-emerald-400 border-l-2 border-emerald-500'
                        : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                    }
                  `}
                >
                  <span
                    className={`flex-shrink-0 ${
                      isActive(item.path) ? 'text-emerald-400' : 'text-gray-500 group-hover:text-gray-300'
                    }`}
                  >
                    {item.icon}
                  </span>
                  {!isCollapsed && (
                    <span className="font-medium whitespace-nowrap">{item.name}</span>
                  )}
                  {/* Tooltip for collapsed state */}
                  {isCollapsed && (
                    <div className="absolute left-full ml-2 px-2 py-1 bg-gray-800 text-white text-sm rounded-md opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all whitespace-nowrap z-50">
                      {item.name}
                    </div>
                  )}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-gray-800">
          {!isCollapsed ? (
            <div className="text-xs text-gray-500 text-center">
              <span className="block">WSB Snake Trading</span>
              <span className="block mt-1">v1.0.0</span>
            </div>
          ) : (
            <div className="w-2 h-2 rounded-full bg-emerald-500 mx-auto" title="System Online" />
          )}
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
