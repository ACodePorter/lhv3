import React, { useState } from 'react';
import { Layout, Menu } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  DatabaseOutlined,
  CodeOutlined,
  LineChartOutlined,
  SettingOutlined,
  HistoryOutlined,
  RobotOutlined,
} from '@ant-design/icons';

const { Sider } = Layout;

const AppSider: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: '首页',
    },
    {
      key: '/data',
      icon: <DatabaseOutlined />,
      label: '数据管理',
    },
    {
      key: '/strategy',
      icon: <CodeOutlined />,
      label: '策略编辑',
    },
    {
      key: '/ai-investment',
      icon: <RobotOutlined />,
      label: 'AI实时投资跑数',
    },
    {
      key: 'backtest',
      icon: <LineChartOutlined />,
      label: '回测管理',
      children: [
        {
          key: '/backtest',
          icon: <LineChartOutlined />,
          label: '回测分析',
        },
        {
          key: '/backtest/history',
          icon: <HistoryOutlined />,
          label: '回测历史',
        },
      ],
    },
    {
      key: '/optimization',
      icon: <SettingOutlined />,
      label: '参数优化',
    },
  ];

  const handleMenuClick = (e: { key: string }) => {
    navigate(e.key);
  };

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={(value) => setCollapsed(value)}
      style={{ background: '#fff' }}
      width={200}
    >
      <div className="logo" style={{ visibility: collapsed ? 'hidden' : 'visible' }}>
        量化交易系统
      </div>
      <Menu
        theme="light"
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={handleMenuClick}
      />
    </Sider>
  );
};

export default AppSider; 
