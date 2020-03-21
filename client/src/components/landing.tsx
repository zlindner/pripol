import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import Search from './search';
import Loader from './loader';
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
    const [url, setURL] = useState('');
    const [loading, setLoading] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);

    const onLoad = () => {
        setLoading(true);

        axios
            .post('/policy/load', { url })
            .then(res => {
                setLoading(false);
                onAnalyze(res.data.policy);
            })
            .catch(err => {
                setLoading(false);
                console.error(err);
            });
    };

    const onAnalyze = (policy: string[]) => {
        setAnalyzing(true);

        axios
            .post('/model/predict', { policy })
            .then(res => {
                console.log(res);
                setAnalyzing(false);
            })
            .catch(err => {
                setAnalyzing(false);
                console.error(err);
            });
    };

    return (
        <Container>
            <Search setURL={setURL} onLoad={onLoad} />

            {(loading || analyzing) && <Loader loading={loading} analyzing={analyzing} url={url} />}
        </Container>
    );
};

export default Landing;
