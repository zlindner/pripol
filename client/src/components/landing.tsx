import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import Search from './search';
import Analysis from './analysis';

const Container = styled.div`
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
`;

const Landing = () => {
    const [loading, setLoading] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);

    const onLoad = () => {};

    const onAnalyze = (policy: string[]) => {
        setAnalyzing(true);

        axios
            .post('/model/predict', { policy })
            .then(res => {
                console.log(res);
            })
            .catch(err => console.error);
    };

    return (
        <Container>
            <Search onStartAnalysis={onStartAnalysis} />

            {analyzing && <Analysis />}
        </Container>
    );
};

export default Landing;
