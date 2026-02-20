import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider, Layout, App as AntdApp } from 'antd';
import enUS from 'antd/locale/en_US';
import { theme } from './theme';

import AppHeader from './components/layout/AppHeader';
import AppSider from './components/layout/AppSider';
import Dashboard from './pages/Dashboard';
import DataManagement from './pages/DataManagement';
import StrategyBuilder from './pages/StrategyBuilder';
import Backtest from './pages/Backtest';
import StrategyOptimization from './pages/StrategyOptimization';
import AiInvestment from './pages/AiInvestment';
import StrategyList from './pages/StrategyList';
import StrategyEditor from './pages/StrategyEditor';
import BacktestHistory from './pages/BacktestHistory';
import BacktestHistoryDetail from './pages/BacktestHistoryDetail';

import './App.css';

const { Content } = Layout;

const App: React.FC = () => {
  return (
    <ConfigProvider locale={enUS} theme={theme}>
      <AntdApp>
        <Router
          future={{
            v7_startTransition: true,
            v7_relativeSplatPath: true
          }}
        >
          <Layout style={{ minHeight: '100vh' }}>
            <AppHeader />
            <Layout>
              <AppSider />
              <Layout style={{ padding: '24px' }}>
                <Content
                  style={{
                    padding: 24,
                    margin: 0,
                    minHeight: 280,
                    background: '#fff',
                    borderRadius: '8px',
                  }}
                >
                  <Routes>
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/data" element={<DataManagement />} />
                    <Route path="/strategy" element={<StrategyList />} />
                    <Route path="/strategy/edit" element={<StrategyEditor />} />
                    <Route path="/strategy/edit/:id" element={<StrategyEditor />} />
                    <Route path="/strategy/builder" element={<StrategyBuilder />} />
                    <Route path="/backtest" element={<Backtest />} />
                    <Route path="/backtest/history" element={<BacktestHistory />} />
                    <Route path="/backtest-history/:id" element={<BacktestHistoryDetail />} />
                    <Route path="/optimization" element={<StrategyOptimization />} />
                    <Route path="/ai-investment" element={<AiInvestment />} />
                  </Routes>
                </Content>
              </Layout>
            </Layout>
          </Layout>
        </Router>
      </AntdApp>
    </ConfigProvider>
  );
};

export default App;
